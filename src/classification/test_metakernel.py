#!/usr/bin/env python3
"""
Test script for MetaKernel implementation
This script performs unit tests and integration tests to verify the correctness
of the conditional normalizing flows and MetaKernel inference networks.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(__file__))

from normalizing_flows import create_conditional_normalizing_flow, CouplingLayer
from inference import create_metakernel_inference
from utilities import sample_normal, dense_layer

def test_coupling_layer():
    """Test the coupling layer implementation"""
    print("Testing CouplingLayer...")
    
    tf.reset_default_graph()
    
    # Test parameters
    batch_size = 4
    z_dim = 8
    context_dim = 16
    hidden_size = 32
    
    # Create placeholders
    z = tf.placeholder(tf.float32, [batch_size, z_dim])
    context = tf.placeholder(tf.float32, [batch_size, context_dim])
    mask = np.zeros(z_dim, dtype=np.float32)
    mask[::2] = 1.0  # Alternate masking
    
    # Create coupling layer
    coupling = CouplingLayer(hidden_size, "test_coupling")
    
    # Forward transformation
    z_transformed, log_det = coupling(z, context, mask)
    
    # Test shapes
    assert z_transformed.get_shape().as_list() == [batch_size, z_dim]
    assert log_det.get_shape().as_list() == [batch_size]
    
    # Test with session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Generate test data
        z_val = np.random.normal(0, 1, [batch_size, z_dim])
        context_val = np.random.normal(0, 1, [batch_size, context_dim])
        
        z_trans_val, log_det_val = sess.run(
            [z_transformed, log_det],
            feed_dict={z: z_val, context: context_val}
        )
        
        # Check that masked dimensions are preserved
        np.testing.assert_allclose(
            z_trans_val[:, ::2],  # masked dimensions
            z_val[:, ::2],        # should be unchanged
            rtol=1e-6
        )
        
        # Check that log determinant is finite
        assert np.all(np.isfinite(log_det_val))
        
    print("✓ CouplingLayer test passed")


def test_conditional_normalizing_flow():
    """Test the conditional normalizing flow implementation"""
    print("Testing ConditionalNormalizingFlow...")
    
    tf.reset_default_graph()
    
    # Test parameters
    batch_size = 4
    z_dim = 8
    context_dim = 16
    num_layers = 3
    hidden_size = 32
    
    # Create flow
    flow = create_conditional_normalizing_flow(num_layers, hidden_size, z_dim)
    
    # Create placeholders
    z0 = tf.placeholder(tf.float32, [batch_size, z_dim])
    context = tf.placeholder(tf.float32, [batch_size, context_dim])
    
    # Forward transformation
    zK, log_det_forward = flow.forward(z0, context)
    
    # Inverse transformation
    z0_reconstructed, log_det_inverse = flow.inverse(zK, context, reuse=True)
    
    # Log probability
    log_prob = flow.log_prob(zK, context, reuse=True)
    
    # Test shapes
    assert zK.get_shape().as_list() == [batch_size, z_dim]
    assert z0_reconstructed.get_shape().as_list() == [batch_size, z_dim]
    assert log_det_forward.get_shape().as_list() == [batch_size]
    assert log_det_inverse.get_shape().as_list() == [batch_size]
    assert log_prob.get_shape().as_list() == [batch_size]
    
    # Test with session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Generate test data
        z0_val = np.random.normal(0, 1, [batch_size, z_dim])
        context_val = np.random.normal(0, 1, [batch_size, context_dim])
        
        zK_val, z0_recon_val, log_det_fwd_val, log_det_inv_val, log_prob_val = sess.run(
            [zK, z0_reconstructed, log_det_forward, log_det_inverse, log_prob],
            feed_dict={z0: z0_val, context: context_val}
        )
        
        # Test invertibility (forward then inverse should recover original)
        np.testing.assert_allclose(z0_val, z0_recon_val, rtol=1e-4, atol=1e-4)
        
        # Test that forward and inverse log determinants are negatives of each other
        np.testing.assert_allclose(log_det_fwd_val, -log_det_inv_val, rtol=1e-4, atol=1e-4)
        
        # Test that log probabilities are finite
        assert np.all(np.isfinite(log_prob_val))
        
    print("✓ ConditionalNormalizingFlow test passed")


def test_metakernel_inference():
    """Test the MetaKernel inference network"""
    print("Testing MetaKernelInference...")
    
    tf.reset_default_graph()
    
    # Test parameters
    batch_size = 4
    d_theta = 64
    d_rn_f = 128
    num_flow_layers = 2
    flow_hidden_size = 32
    
    # Create inference network
    metakernel_inference = create_metakernel_inference(
        d_theta, d_rn_f, num_flow_layers, flow_hidden_size
    )
    
    # Create placeholders
    context_repr = tf.placeholder(tf.float32, [batch_size, d_theta])
    query_repr = tf.placeholder(tf.float32, [batch_size, d_theta])
    
    # Infer posterior parameters
    posterior_params = metakernel_inference.infer_posterior_parameters(
        context_repr, query_repr
    )
    
    # Sample random features
    flow_samples = metakernel_inference.sample_random_features(
        posterior_params, num_samples=1, use_flow=True
    )
    traditional_samples = metakernel_inference.sample_random_features(
        posterior_params, num_samples=1, use_flow=False
    )
    
    # Compute KL divergence
    kl_div_base = metakernel_inference.compute_kl_divergence(
        posterior_params, use_flow=False
    )
    kl_div_flow = metakernel_inference.compute_kl_divergence(
        posterior_params, use_flow=True
    )
    
    # Test shapes
    assert flow_samples.get_shape().as_list() == [batch_size, d_theta]
    assert traditional_samples.get_shape().as_list()[0] == 1  # sample dimension
    assert traditional_samples.get_shape().as_list()[1] == batch_size
    assert traditional_samples.get_shape().as_list()[2] == d_theta
    
    # Test with session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Generate test data
        context_val = np.random.normal(0, 1, [batch_size, d_theta])
        query_val = np.random.normal(0, 1, [batch_size, d_theta])
        
        flow_samples_val, trad_samples_val, kl_base_val, kl_flow_val = sess.run(
            [flow_samples, traditional_samples, kl_div_base, kl_div_flow],
            feed_dict={context_repr: context_val, query_repr: query_val}
        )
        
        # Test that samples are finite
        assert np.all(np.isfinite(flow_samples_val))
        assert np.all(np.isfinite(trad_samples_val))
        
        # Test that KL divergences are finite and positive
        assert np.all(np.isfinite(kl_base_val)) and np.all(kl_base_val >= 0)
        assert np.all(np.isfinite(kl_flow_val)) and np.all(kl_flow_val >= 0)
        
        # Test that flow and traditional samples are different (with high probability)
        flow_mean = np.mean(flow_samples_val)
        trad_mean = np.mean(trad_samples_val)
        assert abs(flow_mean - trad_mean) > 0.01  # Should be different distributions
        
    print("✓ MetaKernelInference test passed")


def test_integration():
    """Integration test with a simplified classification task"""
    print("Testing integration with simplified task...")
    
    tf.reset_default_graph()
    
    # Simplified task parameters
    batch_size = 2
    n_support = 3
    n_query = 2
    d_theta = 32
    d_rn_f = 64
    n_way = 2
    
    # Create inference network
    metakernel_inference = create_metakernel_inference(d_theta, d_rn_f)
    
    # Simulate features and labels
    support_features = tf.placeholder(tf.float32, [batch_size, n_support, d_theta])
    query_features = tf.placeholder(tf.float32, [batch_size, n_query, d_theta])
    support_labels = tf.placeholder(tf.float32, [batch_size, n_support, n_way])
    
    # Simulate context and query representations
    context_repr = tf.reduce_mean(support_features, axis=1)  # Simple averaging
    query_repr = tf.reduce_mean(query_features, axis=1)
    
    # Infer posterior and sample features
    posterior_params = metakernel_inference.infer_posterior_parameters(
        context_repr, query_repr
    )
    
    flow_features = metakernel_inference.sample_random_features(
        posterior_params, use_flow=True
    )
    
    # Simulate kernel computation
    from utilities import rand_features, dotp_kernel
    bias = tf.constant(np.random.uniform(0, 2*np.pi, [d_rn_f, 1]), dtype=tf.float32)
    
    # Compute random features for support and query
    support_rf = rand_features(flow_features, tf.transpose(support_features, [0, 2, 1]), bias)
    query_rf = rand_features(flow_features, tf.transpose(query_features, [0, 2, 1]), bias)
    
    # Test with session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Generate test data
        support_feat_val = np.random.normal(0, 1, [batch_size, n_support, d_theta])
        query_feat_val = np.random.normal(0, 1, [batch_size, n_query, d_theta])
        support_lab_val = np.eye(n_way)[np.random.randint(0, n_way, [batch_size, n_support])]
        
        support_rf_val, query_rf_val = sess.run(
            [support_rf, query_rf],
            feed_dict={
                support_features: support_feat_val,
                query_features: query_feat_val,
                support_labels: support_lab_val
            }
        )
        
        # Test that random features are computed correctly
        assert support_rf_val.shape == (d_rn_f, batch_size * n_support)
        assert query_rf_val.shape == (d_rn_f, batch_size * n_query)
        assert np.all(np.isfinite(support_rf_val))
        assert np.all(np.isfinite(query_rf_val))
        
    print("✓ Integration test passed")


def run_all_tests():
    """Run all tests"""
    print("Running MetaKernel implementation tests...")
    print("=" * 50)
    
    try:
        test_coupling_layer()
        test_conditional_normalizing_flow()
        test_metakernel_inference()
        test_integration()
        
        print("=" * 50)
        print("✓ All tests passed successfully!")
        print("\nMetaKernel implementation is working correctly.")
        print("You can now run the full training script:")
        print("python run_metakernel_classifier.py --dataset miniImageNet --iterations 1000")
        
    except Exception as e:
        print("=" * 50)
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests() 