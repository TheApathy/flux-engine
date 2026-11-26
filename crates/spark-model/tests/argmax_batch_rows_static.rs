// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source contract for one-launch BF16 batched greedy argmax.

const DISPATCH: &str = include_str!("../src/model/trait_impl/meta.rs");
const WRAPPER: &str = include_str!("../src/layers/ops/sampling.rs");
const KERNEL: &str = include_str!("../../../kernels/gb10/common/argmax_bf16.cu");
const MODEL_INIT: &str = include_str!("../src/model/impl_a1.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index..]
        .find(end)
        .map(|offset| start_index + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

#[test]
fn batch_dispatch_uses_one_row_parallel_launch_with_scalar_fallback() {
    let dispatch = compact(section(
        DISPATCH,
        "pub(super) fn argmax_batch_dispatch(",
        "pub(super) fn hidden_after_norm_dispatch(",
    ));

    let guard = dispatch
        .find("ifn>1&&self.argmax_rows_kernel.0!=0{")
        .expect("row-parallel availability guard");
    let rows = dispatch
        .find("ops::argmax_bf16_rows(self.gpu.as_ref(),self.argmax_rows_kernel,logits_ptr,out_ptr,vasu32,vasu32,nasu32,stream,)?")
        .expect("one row-parallel launch");
    let fallback = dispatch
        .find("foriin0..n{")
        .expect("scalar compatibility fallback");
    let copy = dispatch
        .find("self.gpu.copy_d2h(out_ptr,&mutbuf)?")
        .expect("single batched token readback");

    assert!(guard < rows && rows < fallback && fallback < copy);
    assert_eq!(dispatch.matches("ops::argmax_bf16_rows(").count(), 1);
    assert_eq!(dispatch.matches("copy_d2h(").count(), 1);
}

#[test]
fn row_kernel_is_loaded_optionally_and_scalar_c1_remains_available() {
    let init = compact(MODEL_INIT);
    assert!(init.contains(
        "letargmax_rows_kernel=crate::layers::try_kernel(gpu.as_ref(),\"argmax\",\"argmax_bf16_rows\");"
    ));

    let dispatch = compact(section(
        DISPATCH,
        "pub(super) fn argmax_batch_dispatch(",
        "pub(super) fn hidden_after_norm_dispatch(",
    ));
    assert!(dispatch.contains("ifn>1&&self.argmax_rows_kernel.0!=0{"));
    assert!(dispatch.contains(
        "ops::argmax_bf16(self.gpu.as_ref(),self.argmax_kernel,logits_i,out_i,vasu32,stream,)?"
    ));
}

#[test]
fn row_wrapper_launches_one_block_per_tightly_packed_vocab_row() {
    let wrapper = compact(section(
        WRAPPER,
        "pub fn argmax_bf16_rows(",
        "/// GPU-side argmax over `base_logits + bias`",
    ));
    assert!(wrapper.contains("rows>0&&row_stride>=vocab_size"));
    assert!(wrapper.contains(".grid([rows,1,1]).block([1024,1,1])"));
    assert!(wrapper.contains(
        ".arg_ptr(logits).arg_ptr(out).arg_u32(vocab_size).arg_u32(row_stride).launch(stream)"
    ));
}

#[test]
fn row_kernel_preserves_scalar_lowest_id_winner_per_row() {
    let row_kernel = compact(
        KERNEL
            .split("extern \"C\" __global__ void __launch_bounds__(1024) argmax_bf16_rows(")
            .nth(1)
            .expect("row-parallel kernel"),
    );
    assert!(row_kernel.contains(
        "const__nv_bfloat16*__restrict__x=logits+(unsignedlonglong)blockIdx.x*row_stride;"
    ));
    assert!(row_kernel.contains("floatbest=-CUDART_INF_F;"));
    assert!(row_kernel.contains("unsignedintbest_idx=0xFFFFFFFFu;"));
    assert!(row_kernel.contains("if(v>best||(v==best&&i<best_idx)){"));
    assert!(row_kernel.contains(
        "if(s_val[tid+s]>s_val[tid]||(s_val[tid+s]==s_val[tid]&&s_idx[tid+s]<s_idx[tid])){"
    ));
    assert!(row_kernel.contains("out[blockIdx.x]=s_idx[0];"));
}
