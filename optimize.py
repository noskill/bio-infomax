

def optimize1(loss, infomax, opt_encoder, opt_global_discriminator, opt_local_discriminator, opt_prior_discriminator):
    param_encoder = list(infomax.feature_network.parameters()) +\
            list(infomax.aggregator_network.parameters())

    # optimize resnet + encoder
    infomax.zero_grad()
    opt_encoder.zero_grad()
    # optimize model with respect to local discriminator
    loss['local_encoder_loss'].backward(inputs=param_encoder, retain_graph=True)

    loss['grad_resnet_l'] = resnet341[0].weight.grad.abs().max()

    # optimize model with respect to global discriminator
    loss['global_encoder_loss'].backward(inputs=param_encoder, retain_graph=True)
    loss['grad_resnet_g'] = resnet341[0].weight.grad.abs().max()

    # optimize model with respect to prior discriminator
    loss['prior_encoder_loss'].backward(inputs=param_encoder, retain_graph=True)
    loss['grad_resnet_p'] = resnet341[0].weight.grad.abs().max()
    opt_encoder.step()

    # optimize global discriminator
    infomax.zero_grad()
    opt_global_discriminator.zero_grad()
    loss['global_discriminator_loss'].backward(inputs=list(global_loss.parameters()),
            retain_graph=True)
    loss['grad_global_disc'] = global_loss.layer0.weight.grad.abs().max()
    opt_global_discriminator.step()

    # optimize local discriminator
    infomax.zero_grad()
    opt_local_discriminator.zero_grad()
    loss['local_discriminator_loss'].backward(inputs=list(local_loss.parameters()))
    loss['grad_local_disc'] = local_loss.layer0.weight.grad.abs().max()
    opt_local_discriminator.step()

    # optimize prior discriminator
    opt_prior_discriminator.zero_grad()
    loss['prior_discriminator_loss'].backward(inputs=list(prior_disc.parameters()))
    opt_prior_discriminator.step()


def optimize2(loss, infomax, opt_encoder, opt_global_discriminator, opt_local_discriminator, opt_prior_discriminator):
    param_encoder = list(infomax.feature_network.parameters()) +\
            list(infomax.aggregator_network.parameters())
    infomax.zero_grad()
    losses = infomax.losses
    opt_encoder.zero_grad()
    loss_encoder = loss['local_encoder_loss'] + \
            loss['global_encoder_loss'] + loss['prior_encoder_loss']
    loss_encoder.backward(inputs=param_encoder, retain_graph=True)
    loss['sum_encoder'] = loss_encoder.detach()
    loss['grad_resnet_p'] = infomax.feature_network[0].weight.grad.abs().max()
    opt_encoder.step()

    # optimize global discriminator
    infomax.zero_grad()
    opt_global_discriminator.zero_grad()
    loss['global_discriminator_loss'].backward(inputs=list(losses['global_loss'].parameters()),
            retain_graph=True)
    loss['grad_global_disc'] = losses['global_loss'].layer0.weight.grad.abs().max()
    opt_global_discriminator.step()

    # optimize local discriminator
    infomax.zero_grad()
    opt_local_discriminator.zero_grad()
    loss['local_discriminator_loss'].backward(inputs=list(losses['local_loss'].parameters()))
    loss['grad_local_disc'] = losses['local_loss'].layer0.weight.grad.abs().max()
    opt_local_discriminator.step()

    # optimize prior discriminator
    opt_prior_discriminator.zero_grad()
    loss['prior_discriminator_loss'].backward(inputs=list(losses['prior_loss'].parameters()))
    opt_prior_discriminator.step()

