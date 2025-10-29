
import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module): #Not final
#     def __init__(self, alpha=0.35, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
        

# class FocalLoss1(nn.Module): #Not final
#     def __init__(self, alpha=0.35, gamma=2, logits=False, reduction='none'):
#         super(FocalLoss1, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduction

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        
#         return torch.mean(F_loss)
        
        

# class FocalLoss2(nn.Module): #Not final
#     def __init__(self, alpha=0.25, logits=False, reduction='mean'):
#         super(FocalLoss2, self).__init__()
#         self.alpha = alpha
    
#         self.logits = logits
#         self.reduction = reduction

#     def forward(self, inputs, targets):
        
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         # print (pt.shape)
#         gamma = (1+4*pt)
#         # gamma = 2
#         # print (gamma.shape)
#         F_loss = self.alpha * (1-pt)**gamma * BCE_loss
#         # print (F_loss.shape)

#         if self.reduction:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
        


# class FocalLoss3(nn.Module):       ################## resulted in NAN values !!!!!!!!!!!
#     def __init__(self, alpha=0.25, logits=False, reduction='none'):
#         super(FocalLoss3, self).__init__()
#         self.alpha = alpha   
#         self.logits = logits
#         self.reduction = reduction
#         self.relu = nn.ReLU()
   
#     def forward(self, y_pred, targets):
#         gamma = 2
#         weight_a = self.alpha * (1 - y_pred) ** gamma * targets
#         weight_b = (1 - self.alpha) * y_pred ** gamma * (1 - targets)
#         y_pred = torch.clamp(y_pred, torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps)
#         logits = torch.log(y_pred / (1 - y_pred))
        

#         F_loss = (torch.log1p(torch.exp(-torch.abs(logits))) + self.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
#         return torch.mean(F_loss) 


class FocalLoss4(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss4, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get predicted probabilities
        probs = torch.sigmoid(logits)
        
        # Compute the focal loss components
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# class DiceLoss(nn.Module): #Not final
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)
         
#         probs = probs.view(-1)      # Flatten the tensors to 1D
#         targets = targets.view(-1)  
   
#         # Calculate the intersection and union
#         intersection = (probs * targets).sum()
#         union = probs.sum() + targets.sum()
        
#         dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)    # Compute the Dice coefficient
        
#         dice_loss = 1 - dice_coeff      # Calculate Dice loss (1 - Dice coefficient)
        
#         return dice_loss


# class DiceLoss2(nn.Module): #Not final
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss2, self).__init__()
#         self.smooth = smooth

#     def forward(self, y_true, y_pred):
  
#         y_true_f = torch.flatten(y_true)
#         y_pred_f = torch.flatten(y_pred)
#         intersection = torch.sum(y_true_f * y_pred_f)
#         Dice_coef = (2. * intersection + self.smooth) / (
#                 torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
        
#         return 1- Dice_coef
    
class DiceLoss3(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss3, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # Apply sigmoid to logits to get predicted probabilities
        # probs = torch.sigmoid(logits)
        
        # Flatten the tensors to 1D
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # Compute the Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate Dice loss (1 - Dice coefficient)
        dice_loss = 1 - dice_coeff
        
        return dice_loss


class LogCosh3(nn.Module):
    def __init__(self, smooth=1e-6):
        super(LogCosh3, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # Apply sigmoid to logits to get predicted probabilities
        # probs = torch.sigmoid(logits)
        
        # Flatten the tensors to 1D
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # Compute the Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate Dice loss (1 - Dice coefficient)
        dice_loss = 1 - dice_coeff
        
        return torch.log(torch.cosh(dice_loss))
    


    
# class LogCosh2(nn.Module): #Not final
#     def __init__(self, smooth=1e-6):
#         super(LogCosh2, self).__init__()
#         self.smooth = smooth

#     def forward(self, probs, targets):
#         # Apply sigmoid to logits to get predicted probabilities
#         # probs = torch.sigmoid(logits)
        
#         # Flatten the tensors to 1D
#         probs = probs.view(-1)
#         targets = targets.view(-1)
        
#         # Calculate the intersection and union
#         intersection = (probs * targets).sum()
#         union = probs.sum() + targets.sum()
        
#         # Compute the Dice coefficient
#         dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
#         # Calculate Dice loss (1 - Dice coefficient)
#         dice_loss = 1 - dice_coeff
        
#         return torch.log(torch.cosh(dice_loss))


class LogCosh(nn.Module): #Not final
    def __init__(self, smooth=1e-6):
        super(LogCosh, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
  
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        Dice_coef = (2. * intersection + self.smooth) / (
                torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
        DiceLoss = 1- Dice_coef
        return torch.log(torch.cosh(DiceLoss))


# class BCE_DICE(nn.Module):   #Not final
#     def __init__(self, smooth=1e-6):
#         super(BCE_DICE, self).__init__()
#         self.smooth = smooth
#         self.bcel = nn.BCELoss()

#     def forward(self, y_true, y_pred):
        
                
#         y_true_f = torch.flatten(y_true)
#         y_pred_f = torch.flatten(y_pred)
#         intersection = torch.sum(y_true_f * y_pred_f)
#         Dice_coef = (2. * intersection + self.smooth) / (
#                 torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
#         DiceLoss = 1- Dice_coef

#         bcel = self.bcel(y_true, y_pred)
        
#         return (bcel + DiceLoss)/2


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bceLoss = nn.BCELoss()

    def forward(self, probs, targets):
        
        bce_loss = self.bceLoss(probs, targets)
        
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff
        
        loss = (bce_loss + dice_loss)/2
        
        return loss



class TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
  
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        Tversky_Index = (true_pos + self.smooth) / (true_pos + alpha * false_neg + (
                1 - alpha) * false_pos + self.smooth)
        return 1-Tversky_Index
    

class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1e-6, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, y_true, y_pred):
  
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        Tversky_Index = (true_pos + self.smooth) / (true_pos + alpha * false_neg + (
                1 - alpha) * false_pos + self.smooth)
        return (1-Tversky_Index)**self.gamma
    

class WeightedBCELoss(nn.Module): #give NaN results
    def __init__(self, pos_weight=0.7, neg_weight=0.3):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, probs, targets):
        # Apply sigmoid to logits to get predicted probabilities
        # probs = torch.sigmoid(logits)
        
        # Calculate weighted BCE loss
        loss = -self.pos_weight * targets * torch.log(probs) - self.neg_weight * (1 - targets) * torch.log(1 - probs)
        return loss.mean()


if __name__ == "__main__":

    # x1 = torch.rand(1,1,256,256)
    # x2 = torch.rand(1,1,256,256)
    # model = FocalLoss2()
    # loss = model (x1,x2)
    # # print ("loss shape: ", loss.shape)
    # print (loss)



    # Example usage
    logits = torch.tensor([0.8, 0.3, 0.6, 0.9], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
    print ("------------------")
    # Instantiate the DiceLoss class and compute the loss
    criterion1 = DiceLoss2()
    loss1 = criterion1(logits, targets)

    print(f'Dice Loss2: {loss1}') #.item()}')

    print ("------------------")

    criterion2 = LogCosh()
    loss2 = criterion2(logits, targets)

    print(f'LogCosh Loss2: {loss2}') #.item()}')
    
    print ("------------------")
   
    
    criterion3 = BCE_DICE()
    loss3 = criterion3(logits, targets)

    print(f'BCE_DICE Loss3: {loss3}') #.item()}')
    
    print ("------------------")
    
    
    criterion4 = TverskyLoss()
    loss4 = criterion4(logits, targets)

    print(f'Tversky Loss4: {loss4}') #.item()}')


    print ("------------------")

    criterion5 = FocalTverskyLoss()
    loss5 = criterion5(logits, targets)

    print(f'FocalTversky Loss5: {loss5}') #.item()}')

    print ("------------------")
