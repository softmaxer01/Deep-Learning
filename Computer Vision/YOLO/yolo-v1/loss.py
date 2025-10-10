import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, grd=7, box=5, cls=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.grd = grd
        self.box = box
        self.cls = cls
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,pred,tar):
        pred = pred.reshape(-1,self.grd,self.grd,self.cls+self.box*5)

        iou_b1 = intersection_over_union(pred[...,21:25],tar[...,21:25])
        iou_b2 = intersection_over_union(pred[...,26:30],tar[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        iou_maxes,bestbox = torch.max(ious,dim = 0)
        exists_box = tar[...,20].unsqueeze(3) # identity


        # box loss
        box_prediction = exists_box*(bestbox*pred[...,26:30]+(1-bestbox)*pred[...,21:25])
        box_targets = exists_box*tar[...,21:25]

        box_prediction[...,2:4] = torch.sign(torch.sqrt(torch.abs(box_prediction[...,2:4]+1e-6)))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(
            torch.flatten(box_prediction,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2)
        )


        #object loss
        pred_box = (bestbox*pred[...,25:26]+(1-bestbox)*pred[...,20:21])

        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*tar[...,20:21])
        )

        # for no object loss
        no_obj_loss = self.mse(
            torch.flatten((1-exists_box)*pred[...,20:21],start_dim=1),
            torch.flatten((1-exists_box)*tar[...,20:21],start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1-exists_box)*pred[...,25:26],start_dim=1),
            torch.flatten((1-exists_box)*tar[...,20:21],start_dim=1)
        )

        cls_loss = self.mse(
            torch.flatten(exists_box*pred[...:20],end_dim=-2),
            torch.flatten(exists_box*tar[...:20],end_dim=-2),           
        )

        loss = (self.lambda_coord*box_loss
                +object_loss
                +self.lambda_noobj*no_obj_loss
                +cls_loss
                )
        return loss