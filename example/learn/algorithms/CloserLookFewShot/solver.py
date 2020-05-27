import torch
import backbone
import ubelt as ub


def get_model(model_name):
    if model_name == "Conv4":
        return backbone.Conv4
    elif model_name == "Conv4S":
        return backbone.Conv4S
    elif model_name == "Conv6":
        return backbone.Conv6
    elif model_name == "ResNet10":
        return backbone.ResNet10
    elif model_name == "ResNet18":
        return backbone.ResNet18
    elif model_name == "ResNet34":
        return backbone.ResNet34
    elif model_name == "ResNet50":
        return backbone.ResNet50
    elif model_name == "ResNet101":
        return backbone.ResNet101
    else:
        raise NotImplementedError(f"Network name {model_name} not implemented")


def train(arguments, dataloader, model):
    optimizer = torch.optim.Adam(model.parameters())
    max_acc = 0
    acc = 0

    prog = ub.ProgIter(
        range(int(arguments["start_epoch"]), int(arguments["end_epoch"])),
        desc="Training Baseline",
    )
    for epoch in prog:
        model.train()
        # model are called by reference, no need to return
        outStr = model.train_loop(epoch, dataloader, optimizer)
        prog.set_extra(outStr)
        model.eval()

        # if not os.path.isdir(arguments["checkpoint_dir"]):
        #     os.makedirs(arguments["checkpoint_dir"])

        acc = model.test_loop(dataloader)
        # for baseline and baseline++, we don't use validation in default and we
        # let acc = -1, but we allow options to validate with DB index
    #         if acc > max_acc :
    #             print("best model! save...")
    #             max_acc = acc
    #             outfile = os.path.join(
    #                 arguments["checkpoint_dir"], 'best_model.tar')
    #             torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
    #
    # #            if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
    #         if epoch == int(arguments["end_epoch"])-1:
    #             outfile = os.path.join(
    #                 arguments["checkpoint_dir"], '{:d}.tar'.format(epoch))
    #             torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return acc, model
