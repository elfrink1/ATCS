'''Initial setup: We have a base model M_init (e.g. BERT + small MLP) with parameters θ'''

'''For each episode:
Step 1 Sample a support set S and query set Q for the task'''

'''Step 2 Duplicate the base model with deepcopy, the higher library or
by copying the parameters θ to a new instance. The new model is
referred to as Mepisode with parameters θ(0). Make sure the gradients
are zero in the model.'''

'''Step 3 Apply your original base model, M_init, on the support set S and
calculate the prototype-based parameters of the linear layer, γ. Do
not apply torch.no grad() or similar here. We will need gradients
through the init part.'''

'''Step 4 Initialize the output layer parameters φ(0) with the previously 
calculated prototype initialization γ. Thereby, detach the initialization 
so that the computation graph for calculating the prototypes is independent of the inner loop updates'''

'''Step 5 Take k inner loop steps on the support set S with your episode model, Mepisode, 
including the output parameters. Your final parameters are θ(k) and φ(k)'''

'''Step 6 Replace φ(k) with φ(k) = γ + detach(φ(k) − γ)'''


'''This trick adds the original prototypes back to the computation graph
without changing the output parameter values. Note that you have
to do this for both weight and bias parameter. In particular, for the
weight parameter, your code could look something like this:
W = 2 * prototypes + (W - 2 * prototypes).detach()'''

'''Step 7 Apply the trained episode model Mepisode on the query set Q, and
calculate the gradients with respect to θ(k) and θ using torch.autograd.grad.'''

'''Step 8 Sum the gradients for θ(k) and θ, and store them in the gradient
placeholder of your original model M_init. In case this is not the first
episode, sum the new gradients with the ones already stored in Minit.
Outer loop update Perform one update step using the gradients stored in
M_init. Afterwards, set them to zero.'''