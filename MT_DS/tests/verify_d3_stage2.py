"""Check M0 backbone equivalence, replayed BN/Dropout, and transfer gradients."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary import make_band
from d3_transfer import BoundaryPrototypeTransfer, signed_donor_scores, labeled_injection_check
from networks.vnet_ds import VNetDS
from networks.vnet_boundary_probe import VNetBoundaryProbe
from networks.vnet_d3_stage2 import VNetD3Stage2
from objectives import supervised_loss, consistency_loss, update_ema
from online_probe import channel_interventions
from probe_objectives import probe_boundary_loss


def hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "code").rglob("*.py"))}


def main():
    torch.set_num_threads(2)
    checks = []
    labels = torch.zeros(2,32,32,32,dtype=torch.long)
    labels[:,8:24,8:24,8:24] = 1
    bands = torch.stack([torch.from_numpy(make_band(x.numpy(),2)) for x in labels])
    target = (bands[:,None]>0).float()
    torch.manual_seed(42)
    plain = VNetDS(1,2,normalization="batchnorm",has_dropout=True).train()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    m0 = VNetD3Stage2(1,2,normalization="batchnorm",has_dropout=True,readout_only=True).train()
    assert torch.equal(rng,torch.get_rng_state())
    for n,v in plain.state_dict().items():
        torch.testing.assert_close(v,m0.state_dict()[n],rtol=0,atol=0)
    teacher_a,teacher_b = copy.deepcopy(plain).requires_grad_(False),copy.deepcopy(m0).requires_grad_(False)
    optim_a = torch.optim.SGD(plain.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
    optim_b = torch.optim.SGD(m0.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
    image_l,image_u = torch.randn(2,1,32,32,32),torch.randn(2,1,32,32,32)
    for step in range(2):
        rng = torch.get_rng_state().clone()
        out_a = plain(image_l,deepsup=True)
        un_a = plain(image_u,deepsup=True)
        with torch.no_grad(): t_a=teacher_a(image_u,deepsup=True)
        sup_a,*_ = supervised_loss(out_a,labels)
        loss_a=sup_a+.3*consistency_loss(un_a,t_a)
        optim_a.zero_grad(set_to_none=True);loss_a.backward()
        optim_a.step();update_ema(plain,teacher_a,.99,step)
        next_rng=torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        out_b,logits,features=m0(image_l,deepsup=True,return_boundary=True)
        un_b=m0(image_u,deepsup=True)
        with torch.no_grad(): t_b=teacher_b(image_u,deepsup=True)
        sup_b,*_=supervised_loss(out_b,labels)
        boundary,_=probe_boundary_loss(logits,target,"d3")
        loss_b=sup_b+.3*consistency_loss(un_b,t_b)+.015*boundary
        optim_b.zero_grad(set_to_none=True);loss_b.backward()
        score=signed_donor_scores(m0,features,target)
        assert score.norm()>0 and m0.boundary_head[0].weight.grad.norm()>0
        for n,p in plain.named_parameters():
            torch.testing.assert_close(p.grad,dict(m0.named_parameters())[n].grad,rtol=0,atol=0)
        optim_b.step();update_ema(m0,teacher_b,.99,step)
        assert torch.equal(next_rng,torch.get_rng_state())
        for n,v in plain.state_dict().items():
            torch.testing.assert_close(v,m0.state_dict()[n],rtol=0,atol=0)
            torch.testing.assert_close(teacher_a.state_dict()[n],teacher_b.state_dict()[n],rtol=0,atol=0)
        assert all(p.grad is None for p in teacher_b.parameters())
    checks.append("M0_active_readout_two_SGD_EMA_steps_exact_Baseline_backbone_BN_RNG_gradients")
    del plain,m0,teacher_a,teacher_b,optim_a,optim_b,out_a,out_b,un_a,un_b,features

    torch.manual_seed(42)
    old=VNetBoundaryProbe(1,2,normalization="batchnorm",has_dropout=True,variant="d3").train()
    rng=torch.get_rng_state().clone()
    torch.manual_seed(42)
    new=VNetD3Stage2(1,2,normalization="batchnorm",has_dropout=True,interaction=True).train()
    assert torch.equal(rng,torch.get_rng_state())
    rng=torch.get_rng_state().clone()
    ref_l,ref_b,_=old(image_l,deepsup=True,return_boundary=True)
    ref_u=old(image_u,deepsup=True)
    reference_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    cur_l,cur_b,features=new(image_l,deepsup=True,return_boundary=True)
    cur_u,context=new.forward_context(image_u)
    assert torch.equal(reference_rng,torch.get_rng_state())
    before_buffers={n:b.clone() for n,b in new.named_buffers()}
    half,full=new.replay_tail(context["half"],context,checkpoint_tail=True)
    torch.testing.assert_close(half,cur_u[-2],rtol=0,atol=0)
    torch.testing.assert_close(full,cur_u[-1],rtol=0,atol=0)
    assert torch.equal(reference_rng,torch.get_rng_state())
    teacher_outputs=[torch.zeros_like(x) for x in ref_u]
    sup_ref,*_=supervised_loss(ref_l,labels)
    sup_new,*_=supervised_loss(cur_l,labels)
    lb_ref,_=probe_boundary_loss(ref_b,target,"d3")
    lb_new,_=probe_boundary_loss(cur_b,target,"d3")
    loss_ref=sup_ref+.015*lb_ref+.3*consistency_loss(ref_u,teacher_outputs)
    loss_new=sup_new+.015*lb_new+.3*.5*(consistency_loss(cur_u,teacher_outputs)+
                        consistency_loss([*cur_u[:2],half,full],teacher_outputs))
    loss_ref.backward();loss_new.backward()
    for n,p in old.named_parameters():
        torch.testing.assert_close(p.grad,dict(new.named_parameters())[n].grad,rtol=2e-5,atol=2e-6)
    for n,b in new.named_buffers():
        torch.testing.assert_close(b,before_buffers[n],rtol=0,atol=0)
    assert torch.equal(reference_rng,torch.get_rng_state())
    assert all(p.grad is None for p in new.transfer.parameters())
    checks.append("zero_residual_checkpointed_tail_exact_logits_cloned_BN_same_Dropout_equivalent_gradients")
    del old,ref_l,ref_u,ref_b,cur_l,cur_b,cur_u,features,context,half,full

    donor=torch.randn(2,32,4,4,4,requires_grad=True)
    receiver=torch.randn(2,32,4,4,4,requires_grad=True)
    scores=torch.arange(32,0,-1,dtype=torch.float32).repeat(2,1).requires_grad_()
    small_band=bands[:,:8,:8,:8].clone()
    small_band.zero_();small_band[:,1:4]=1;small_band[:,4:7]=2
    prediction=torch.zeros(2,2,8,8,8)
    prediction[:,0,:4]=3.;prediction[:,1,4:]=3.
    prediction.requires_grad_()
    for mode in ("random","boundary"):
        new.transfer.zero_grad(set_to_none=True)
        residual,stats,route=new.transfer(donor,small_band,scores,receiver,prediction,
                                          mode=mode,alpha=.1,seed=42,step=5000)
        assert stats["interaction_active_samples"]==2 and stats["interaction_residual_ratio_max"]<=.100001
        mask=torch.zeros(2,32).scatter_(1,torch.tensor(route["indices"]),1.)
        assert (residual*(1-mask)[...,None,None,None]).count_nonzero()==0
        residual.square().sum().backward()
        assert donor.grad is None and scores.grad is None and prediction.grad is None
        assert receiver.grad.norm()>0
        assert all(p.grad is not None and p.grad.norm()>0 for p in new.transfer.parameters())
        zero,_,_=new.transfer(donor,small_band,scores,receiver,prediction,
                              mode=mode,alpha=0.,seed=42,step=5000)
        assert zero.count_nonzero()==0
        empty,stats,_=new.transfer(donor,torch.zeros_like(small_band),scores,receiver,prediction,
                                   mode=mode,alpha=.1,seed=42,step=5000)
        assert empty.count_nonzero()==0 and stats["interaction_valid_donors"]==0
    checks.append("both_routes_sparse_equal_k_norm_cap_empty_skip_donor_teacher_stopgrad_trainable_transfer")

    new.zero_grad(set_to_none=True)
    new.train()
    out_l,edge,features=new(image_l,deepsup=True,return_boundary=True)
    out_u,context=new.forward_context(image_u)
    synthetic_teacher=torch.zeros(2,2,32,32,32)
    synthetic_teacher[:,0,:16]=3.;synthetic_teacher[:,1,16:]=3.
    residual,stats,route=new.transfer(features[1],bands,scores.detach(),context["features"][1],
                                     synthetic_teacher,mode="boundary",alpha=.1,seed=42,step=5000)
    assert stats["interaction_active_samples"]==2
    buffers={n:b.clone() for n,b in new.named_buffers()}
    rng=torch.get_rng_state().clone()
    half,full=new.replay_tail(context["half"]+residual,context)
    assert not torch.equal(full,out_u[-1])
    consistency_loss([*out_u[:2],half,full],[torch.zeros_like(x) for x in out_u]).backward()
    assert new.transfer.output.weight.grad.norm()>0 and new.block_one.conv[0].weight.grad.norm()>0
    for n,b in new.named_buffers(): torch.testing.assert_close(b,buffers[n],rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    gradients={n:p.grad.clone() for n,p in new.named_parameters() if p.grad is not None}
    probe=labeled_injection_check(new,features,labels,bands,scores.detach(),mode="boundary",
                                   alpha=.1,seed=42,step=5000,fraction=.125)
    assert "final_boundary_ce" in probe["actual_delta"]
    for n,p in new.named_parameters():
        if n in gradients: torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    checks.append("active_checkpointed_L2U_updates_receiver_and_transfer_without_extra_BN_RNG_or_diagnostic_grad_updates")
    report=PROJECT/"verification_runs/d3_stage2_v1/unit_results.json"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(dict(status="PASS",checks=checks,source_hashes=hashes()),indent=2)+"\n")
    print(json.dumps(dict(status="PASS",checks=checks,report=str(report)),indent=2))


if __name__=="__main__":
    main()
