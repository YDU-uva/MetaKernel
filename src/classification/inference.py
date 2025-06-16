import tensorflow as tf
from utilities import dense_layer, sample_normal
from normalizing_flows import create_conditional_normalizing_flow


def inference_block(inputs, d_theta, output_units, name):
    """
    Three dense layers in sequence.
    :param inputs: batch of inputs.
    :param d_theta: dimensionality of the intermediate hidden layers.
    :param output_units: dimensionality of the output.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    h = dense_layer(inputs, d_theta, tf.nn.elu, True, name + '1')
    h = dense_layer(h, d_theta, tf.nn.elu, True, name + '2')
    h = dense_layer(h, output_units, None, True, name + '3')
    return h


class MetaKernelInference:
    """
    Enhanced inference network for MetaKernel with conditional normalizing flows
    Based on "MetaKernel: Learning Variational Random Features with Limited Labels"
    """
    
    def __init__(self, d_theta, d_rn_f, num_flow_layers=4, flow_hidden_size=128, name="metakernel_inference"):
        self.d_theta = d_theta
        self.d_rn_f = d_rn_f
        self.num_flow_layers = num_flow_layers
        self.flow_hidden_size = flow_hidden_size
        self.name = name
        
        # Create conditional normalizing flow for enhanced posterior
        self.conditional_flow = create_conditional_normalizing_flow(
            num_layers=num_flow_layers,
            hidden_size=flow_hidden_size,
            z_dim=d_theta,
            name=f"{name}_flow"
        )
    
    def infer_posterior_parameters(self, context_representation, query_representation, reuse=False):
        """
        Infer posterior distribution parameters using both traditional inference and normalizing flows
        
        Args:
            context_representation: Context representation from LSTM [batch_size, d_theta]
            query_representation: Query representation from attention [batch_size, d_theta]
            reuse: Whether to reuse variables
            
        Returns:
            Dictionary containing posterior parameters and flow samples
        """
        with tf.variable_scope(self.name, reuse=reuse):
            # Traditional variational inference for base distribution parameters
            with tf.variable_scope('base_posterior'):
                # Context-based posterior (r distribution)
                r_mu = inference_block(context_representation, self.d_theta, self.d_theta, 'r_mean')
                r_log_var = inference_block(context_representation, self.d_theta, self.d_theta, 'r_log_var')
                
                # Query-based prior (p distribution)  
                p_mu = inference_block(query_representation, self.d_theta, self.d_theta, 'p_mean')
                p_log_var = inference_block(query_representation, self.d_theta, self.d_theta, 'p_log_var')
            
            # Enhanced posterior using conditional normalizing flows
            with tf.variable_scope('flow_posterior'):
                # Use context representation as conditioning information for the flow
                flow_context = tf.concat([context_representation, query_representation], axis=-1)
                flow_context = dense_layer(flow_context, self.flow_hidden_size, tf.nn.relu, True, 'flow_context')
                
                # Sample from base distribution (using r parameters)
                base_samples = sample_normal(r_mu, r_log_var, 1)  # [batch_size, 1, d_theta]
                base_samples = tf.squeeze(base_samples, axis=1)  # [batch_size, d_theta]
                
                # Transform samples through conditional normalizing flow
                flow_samples, flow_log_det = self.conditional_flow.forward(
                    base_samples, flow_context, reuse=reuse
                )
                
                # Compute flow-enhanced log probability
                flow_log_prob = self.conditional_flow.log_prob(
                    flow_samples, flow_context, reuse=True
                )
        
        return {
            'r_mu': r_mu,
            'r_log_var': r_log_var,
            'p_mu': p_mu,
            'p_log_var': p_log_var,
            'flow_samples': flow_samples,
            'flow_log_det': flow_log_det,
            'flow_log_prob': flow_log_prob,
            'flow_context': flow_context
        }
    
    def sample_random_features(self, posterior_params, num_samples=1, use_flow=True, reuse=False):
        """
        Sample random features from the enhanced posterior distribution
        
        Args:
            posterior_params: Dictionary from infer_posterior_parameters
            num_samples: Number of samples to draw
            use_flow: Whether to use flow-enhanced sampling
            reuse: Whether to reuse variables
            
        Returns:
            Random feature samples [batch_size, num_samples, d_theta] or [batch_size, d_theta]
        """
        if use_flow:
            # Use flow-enhanced samples
            if num_samples == 1:
                return posterior_params['flow_samples']
            else:
                # Sample multiple times from the flow
                flow_samples = self.conditional_flow.sample(
                    posterior_params['flow_context'], 
                    num_samples=num_samples, 
                    reuse=True
                )
                return flow_samples
        else:
            # Use traditional sampling from base distribution
            return sample_normal(
                posterior_params['r_mu'], 
                posterior_params['r_log_var'], 
                num_samples
            )
    
    def compute_kl_divergence(self, posterior_params, use_flow=False):
        """
        Compute KL divergence between posterior and prior
        
        Args:
            posterior_params: Dictionary from infer_posterior_parameters
            use_flow: Whether to include flow-based KL term
            
        Returns:
            KL divergence value
        """
        # Base KL divergence between Gaussian distributions
        from utilities import KL_divergence
        base_kl = KL_divergence(
            posterior_params['r_mu'], 
            posterior_params['r_log_var'],
            posterior_params['p_mu'], 
            posterior_params['p_log_var']
        )
        
        if use_flow:
            # Add flow-based regularization term
            # This encourages the flow to not deviate too much from the base distribution
            flow_regularization = -tf.reduce_mean(posterior_params['flow_log_det'])
            return base_kl + 0.1 * flow_regularization  # weight the flow regularization
        else:
            return base_kl


def create_metakernel_inference(d_theta, d_rn_f, num_flow_layers=4, flow_hidden_size=128, name="metakernel_inference"):
    """
    Factory function to create MetaKernel inference network
    
    Args:
        d_theta: Feature dimension
        d_rn_f: Random feature dimension  
        num_flow_layers: Number of coupling layers in normalizing flow
        flow_hidden_size: Hidden size for flow networks
        name: Variable scope name
        
    Returns:
        MetaKernelInference instance
    """
    return MetaKernelInference(d_theta, d_rn_f, num_flow_layers, flow_hidden_size, name)

