
class TestAether(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
    def test_aether_config(self):
        """Test that Aether config can be passed and runs without error."""
        from tensorrt_llm.llmapi import AetherSparseAttentionConfig
        
        aether_config = AetherSparseAttentionConfig(
            block_size=64,
            threshold=0.15,
            use_variance=True,
            use_concentration=True
        )
        
        pytorch_config = dict(
            sparse_attention_config=aether_config
        )
        
        with LLM(self.MODEL_PATH, **pytorch_config) as llm:
            assert isinstance(llm.args.sparse_attention_config, AetherSparseAttentionConfig)
            assert llm.args.sparse_attention_config.algorithm == "aether"
            
            # Simple inference test
            output = llm.generate(["AETHER is"], sampling_params=SamplingParams(max_tokens=10))
            assert len(output[0].outputs[0].token_ids) == 10
