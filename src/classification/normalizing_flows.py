import tensorflow as tf
import numpy as np
from utilities import dense_layer


class CouplingLayer:
    """
    Implementation of conditional normalizing flows based on coupling layers
    Based on the paper "MetaKernel: Learning Variational Random Features with Limited Labels"
    """
    
    def __init__(self, hidden_size, name):
        self.hidden_size = hidden_size
        self.name = name
        
    def __call__(self, z, context, mask, reuse=False):
        """
        Forward pass
        Args:
            z: Input variables [batch_size, dim]
            context: Context information [batch_size, context_dim] 
            mask: Mask specifying which dimensions remain unchanged [dim]
            reuse: Whether to reuse variables
        Returns:
            z_transformed: Transformed variables
            log_det_jacobian: Log determinant of Jacobian
        """
        with tf.variable_scope(self.name, reuse=reuse):
            # Split input
            z_a = z * mask
            z_b = z * (1 - mask)
            
            # Build input to conditional network
            coupling_input = tf.concat([z_a, context], axis=-1)
            
            # Conditional transformation network
            h1 = dense_layer(coupling_input, self.hidden_size, tf.nn.relu, True, 'coupling_h1')
            h2 = dense_layer(h1, self.hidden_size, tf.nn.relu, True, 'coupling_h2')
            
            # Compute transformation parameters
            s = dense_layer(h2, tf.shape(z_b)[-1], None, True, 'coupling_s')  # scaling parameter
            t = dense_layer(h2, tf.shape(z_b)[-1], None, True, 'coupling_t')  # translation parameter
            
            # Apply affine transformation
            s = tf.tanh(s)  # constrain scaling parameter range
            z_b_transformed = z_b * tf.exp(s) + t
            
            # Reconstruct output
            z_transformed = z_a + z_b_transformed * (1 - mask)
            
            # Compute log determinant of Jacobian (only scaling part contributes)
            log_det_jacobian = tf.reduce_sum(s * (1 - mask), axis=-1)
            
            return z_transformed, log_det_jacobian


class ConditionalNormalizingFlow:
    """
    Main implementation of conditional normalizing flows
    Integrates multiple coupling layers to build complex conditional transformations
    """
    
    def __init__(self, num_layers, hidden_size, z_dim, name="conditional_nf"):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.name = name
        
        # Create alternating mask patterns
        self.masks = []
        for i in range(num_layers):
            mask = np.zeros(z_dim, dtype=np.float32)
            mask[i % 2::2] = 1.0  # alternating mask
            self.masks.append(mask)
            
        # Initialize coupling layers
        self.coupling_layers = []
        for i in range(num_layers):
            layer = CouplingLayer(hidden_size, f"{name}_coupling_{i}")
            self.coupling_layers.append(layer)
    
    def forward(self, z0, context, reuse=False):
        """
        Forward transformation: z0 -> zK
        Args:
            z0: Base distribution samples [batch_size, z_dim]
            context: Context information [batch_size, context_dim]
            reuse: Whether to reuse variables
        Returns:
            zK: Transformed samples
            log_det_jacobian: Total log determinant of Jacobian
        """
        z = z0
        total_log_det = 0.0
        
        for i, (layer, mask) in enumerate(zip(self.coupling_layers, self.masks)):
            z, log_det = layer(z, context, mask, reuse=(reuse or i > 0))
            total_log_det += log_det
            
        return z, total_log_det
    
    def inverse(self, zK, context, reuse=True):
        """
        Inverse transformation: zK -> z0 (for density estimation during training)
        Args:
            zK: Transformed variables [batch_size, z_dim]
            context: Context information [batch_size, context_dim]
            reuse: Whether to reuse variables
        Returns:
            z0: Base distribution variables
            log_det_jacobian: Total log determinant of Jacobian
        """
        z = zK
        total_log_det = 0.0
        
        # Apply transformations in reverse order
        for i in reversed(range(self.num_layers)):
            layer = self.coupling_layers[i]
            mask = self.masks[i]
            
            with tf.variable_scope(layer.name, reuse=True):
                # Split input
                z_a = z * mask
                z_b = z * (1 - mask)
                
                # Build input to conditional network
                coupling_input = tf.concat([z_a, context], axis=-1)
                
                # Conditional transformation network (reuse weights)
                h1 = dense_layer(coupling_input, self.hidden_size, tf.nn.relu, True, 'coupling_h1')
                h2 = dense_layer(h1, self.hidden_size, tf.nn.relu, True, 'coupling_h2')
                
                # Compute transformation parameters
                s = dense_layer(h2, tf.shape(z_b)[-1], None, True, 'coupling_s')
                t = dense_layer(h2, tf.shape(z_b)[-1], None, True, 'coupling_t')
                
                # Inverse affine transformation
                s = tf.tanh(s)
                z_b_inverse = (z_b - t) * tf.exp(-s)
                
                # Reconstruct output
                z = z_a + z_b_inverse * (1 - mask)
                
                # Accumulate Jacobian determinant
                total_log_det -= tf.reduce_sum(s * (1 - mask), axis=-1)
        
        return z, total_log_det
    
    def log_prob(self, z, context, reuse=False):
        """
        Compute log probability density of transformed distribution
        Args:
            z: Points in target space [batch_size, z_dim]
            context: Context information [batch_size, context_dim]
            reuse: Whether to reuse variables
        Returns:
            log_prob: Log probability density
        """
        # Inverse transform to base distribution
        z0, log_det = self.inverse(z, context, reuse)
        
        # Log probability of base distribution (standard Gaussian)
        log_prob_base = -0.5 * tf.reduce_sum(z0**2, axis=-1) - 0.5 * self.z_dim * tf.log(2.0 * np.pi)
        
        # Apply change of variables formula
        log_prob = log_prob_base + log_det
        
        return log_prob
    
    def sample(self, context, num_samples=1, reuse=False):
        """
        Sample from conditional distribution
        Args:
            context: Context information [batch_size, context_dim]
            num_samples: Number of samples
            reuse: Whether to reuse variables
        Returns:
            samples: Sampling results [batch_size, num_samples, z_dim]
        """
        batch_size = tf.shape(context)[0]
        
        # Sample from base distribution
        z0 = tf.random_normal([batch_size, num_samples, self.z_dim])
        
        samples_list = []
        for i in range(num_samples):
            z0_i = z0[:, i, :]
            z_i, _ = self.forward(z0_i, context, reuse=(reuse or i > 0))
            samples_list.append(tf.expand_dims(z_i, axis=1))
        
        samples = tf.concat(samples_list, axis=1)
        return samples


def create_conditional_normalizing_flow(num_layers, hidden_size, z_dim, name="conditional_nf"):
    """
    Factory function to create conditional normalizing flow
    Args:
        num_layers: Number of coupling layers
        hidden_size: Hidden layer size
        z_dim: Latent variable dimension
        name: Variable scope name
    Returns:
        ConditionalNormalizingFlow instance
    """
    return ConditionalNormalizingFlow(num_layers, hidden_size, z_dim, name) 