def function(session, optimizer):
#    session.run(optimizer);
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  #if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(loss(model, training_inputs, training_outputs)))
    return;
