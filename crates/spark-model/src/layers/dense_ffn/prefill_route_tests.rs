// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(test)]
mod prefill_ffn_fast_route_tests {
    use super::super::{PrefillFfnFastRoute as Route, prefill_ffn_fast_route};

    #[test]
    fn distinguishes_disabled_ineligible_complete_and_every_missing_dependency() {
        assert_eq!(
            prefill_ffn_fast_route(false, true, true, true, true),
            Route::Disabled
        );
        assert_eq!(
            prefill_ffn_fast_route(true, false, false, false, false),
            Route::Ineligible
        );
        assert_eq!(
            prefill_ffn_fast_route(true, true, true, true, true),
            Route::Complete
        );
        for readiness in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert_eq!(
                prefill_ffn_fast_route(true, true, readiness.0, readiness.1, readiness.2),
                Route::Missing
            );
        }
    }

    #[test]
    fn missing_fails_before_projection_and_marker_follows_completed_down_projection() {
        let source = concat!(include_str!("prefill.rs"), include_str!("policy.rs"));
        let select = source.find("let fast_route =").expect("FAST selector");
        let branch = source[select..]
            .find("if fast_path {")
            .map(|offset| select + offset)
            .expect("FAST projection branch");
        let selection = &source[select..branch];
        assert!(selection.contains("PrefillFfnFastRoute::Missing"));
        assert!(selection.contains("before gate projection"));

        let end = source[branch..]
            .find("// Byte-exact cp.async pipelined path")
            .map(|offset| branch + offset)
            .expect("FAST projection boundary");
        let body = &source[branch..end];
        let down = body
            .rfind("ops::w4a16_gemm_n128_m128(")
            .expect("FAST down projection");
        let marker = body
            .find("log_prefill_ffn_fast_route_once(m, h, inter);")
            .expect("FAST completion marker");
        let returned = body.rfind("return Ok(());").expect("FAST return");
        assert!(down < marker && marker < returned);
        assert!(source.contains(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete"
        ));
    }

    #[test]
    fn prefill_only_copies_waive_m16_for_fast_route_but_not_has_transposed() {
        let source = concat!(
            include_str!("prefill.rs"),
            include_str!("layer.rs"),
            include_str!("e2m1_flashinfer.rs")
        );
        let select = source.find("let fast_route =").expect("FAST selector");
        let selection = &source[select..select + 400];
        assert!(selection.contains("self.w4a16_gemm_t_m16.0 != 0 || self.prefill_t_m128_only,"));

        let has_t = source
            .find("pub fn has_transposed_ffn(&self) -> bool {")
            .expect("has_transposed_ffn");
        let body = &source[has_t..has_t + 300];
        assert!(body.contains("&& self.w4a16_gemm_t_m16.0 != 0"));
        assert!(!body.contains("prefill_t_m128_only"));
        // Leading indent + newline so this test's own literal does not match.
        let setter = "        self.prefill_t_m128_only = true;\n";
        assert_eq!(source.matches(setter).count(), 1);
    }
}

#[cfg(test)]
mod prefill_fused_epilogue_tests {
    use super::super::{FfnActivation, PrefillFusedEpilogueRoute, prefill_fused_epilogue_route};

    #[test]
    fn route_distinguishes_disabled_ineligible_complete_and_missing() {
        assert_eq!(
            prefill_fused_epilogue_route(false, true, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Disabled
        );
        for (activation, m, h, inter) in [
            (FfnActivation::GeLU, 8192, 5120, 17408),
            (FfnActivation::SiLU, 32, 5120, 17408),
            (FfnActivation::SiLU, 8192, 5119, 17408),
            (FfnActivation::SiLU, 8192, 5120, 17407),
        ] {
            assert_eq!(
                prefill_fused_epilogue_route(true, true, true, activation, m, h, inter),
                PrefillFusedEpilogueRoute::Ineligible
            );
        }
        assert_eq!(
            prefill_fused_epilogue_route(true, true, true, FfnActivation::SiLU, 33, 5120, 17408,),
            PrefillFusedEpilogueRoute::Complete
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, false, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, true, false, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
    }

    #[test]
    fn source_fails_before_projection_and_logs_each_engaged_route() {
        let source = concat!(include_str!("prefill.rs"), include_str!("policy.rs"));
        let dispatch_start = source.find("let pipe_requested =").unwrap();
        let dispatch_end = source[dispatch_start..]
            .find("// Baseline M_TILE=64 path")
            .map(|offset| dispatch_start + offset)
            .unwrap();
        let dispatch = &source[dispatch_start..dispatch_end];
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
        ));
        assert!(
            dispatch.contains("Dual is authoritative when both explicit candidate flags are set")
        );
        let log_start = source.find("fn log_prefill_pipe_route_once").unwrap();
        let log_end = source[log_start..]
            .find("enum ExactFfnDispatch")
            .map(|offset| log_start + offset)
            .unwrap();
        let loggers = &source[log_start..log_end];
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_PIPE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED"));

        let dual_launch = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let dual_marker = dispatch
            .find("log_prefill_dual_fused_route_once();")
            .unwrap();
        assert!(dual_launch < dual_marker);
    }

    #[test]
    fn legacy_boolean_silent_fallback_is_absent() {
        let source = concat!(
            include_str!("../dense_ffn.rs"),
            include_str!("policy.rs"),
            include_str!("sm120_coalesced.rs"),
            include_str!("cublaslt_proj.rs"),
            include_str!("e2m1_flashinfer.rs"),
            include_str!("cublaslt_ffn.rs"),
            include_str!("layer.rs"),
            include_str!("decode.rs"),
            include_str!("kgamma.rs"),
            include_str!("prefill.rs"),
        );
        let legacy_name = ["fn use_prefill", "_fused_epilogue("].concat();
        assert!(!source.contains(&legacy_name));
    }

    #[test]
    fn cuda_source_retains_bf16_round_trip_and_in_place_abi() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_silu_mul")
            .expect("fused prefill epilogue kernel is missing");
        let body = &source[start..];
        assert!(body.contains("__nv_bfloat16 up_bf16 = __float2bfloat16(value)"));
        assert!(body.contains("float u = __bfloat162float(up_bf16)"));
        assert!(body.contains("float g = __bfloat162float(gate_in_out[out_idx])"));
        assert!(body.contains("gate_in_out[out_idx] = __float2bfloat16(g * sigmoid_g * u)"));
    }

    #[test]
    fn dual_cuda_source_retains_both_bf16_round_trips_and_independent_weights() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_dual")
            .expect("dual fused prefill kernel is missing");
        let body = &source[start..];
        assert!(body.contains("const unsigned char* __restrict__ gate_packed"));
        assert!(body.contains("const unsigned char* __restrict__ up_packed"));
        assert!(body.contains("__nv_bfloat16* C1"));
        assert!(body.contains("if (fuse_silu != 0)"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(gate_value))"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(up_value))"));
        assert!(body.contains("__float2bfloat16(g * sigmoid_g * u)"));
        assert!(body.contains("C0[out_idx] = __float2bfloat16(g)"));
        assert!(body.contains("C1[out_idx] = __float2bfloat16(u)"));
    }

    #[test]
    fn dual_rust_abi_matches_cuda_and_fused_call_uses_null_second_output() {
        let ops = include_str!("../ops/gemm_dense.rs");
        let start = ops
            .find("pub fn w4a16_gemm_pipe_dual(")
            .expect("dual Rust launcher is missing");
        let body = &ops[start..];
        let ordered = [
            ".arg_ptr(input)",
            ".arg_ptr(gate.weight)",
            ".arg_ptr(gate.weight_scale)",
            ".arg_f32(gate.weight_scale_2)",
            ".arg_ptr(up.weight)",
            ".arg_ptr(up.weight_scale)",
            ".arg_f32(up.weight_scale_2)",
            ".arg_ptr(output_first)",
            ".arg_ptr(output_second)",
            ".arg_u32(u32::from(fuse_silu))",
            ".arg_u32(m)",
            ".arg_u32(n)",
            ".arg_u32(k)",
        ];
        let mut cursor = 0;
        for argument in ordered {
            let relative = body[cursor..]
                .find(argument)
                .unwrap_or_else(|| panic!("missing or misordered dual argument {argument}"));
            cursor += relative + argument.len();
        }

        let host = include_str!("prefill.rs");
        let dispatch_start = host.find("let pipe_requested =").unwrap();
        let dispatch = &host[dispatch_start..];
        let call = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let call_body = &dispatch[call..];
        let null_output = call_body.find("DevicePtr::NULL").unwrap();
        let fused = call_body.find("true,").unwrap();
        let launch_end = call_body.find(")?;").unwrap();
        assert!(null_output < fused && fused < launch_end);
    }
}
