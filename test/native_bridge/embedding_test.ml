(* [torch.ops.aten.embedding.default] against real ATen. The lookup agrees for
   every valid index and for a rank-1 and a rank-2 index tensor; [padding_idx]
   changes nothing in the forward pass (the stored padding row comes back as
   stored). The native engine rejects an index outside [0, V), a negative one
   included, as ATen's [index_select] does. ATen's own rejection cannot be run
   as an oracle here: in this minimal archive the kernel's bounds check fires
   inside a parallel region and aborts the process instead of surfacing the
   error, so the invalid-index behavior is pinned from the native side only. *)

open Helpers

let table () =
  float_tensor [ 5; 3 ]
    (List.init 15 (fun i -> float_of_int ((10 * (i / 3)) + (i mod 3))))

let%expect_test "verify: embedding with a rank-1 index" =
  verify_print ~target:"torch.ops.aten.embedding.default"
    ~bindings:
      [ ("weight", table ()); ("indices", i64_tensor [ 4 ] [ 3L; 0L; 3L; 4L ]) ]
    ~inputs:[ in_tensor "weight"; in_tensor "indices" ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: embedding with a rank-2 index and a padding_idx" =
  verify_print ~target:"torch.ops.aten.embedding.default"
    ~bindings:
      [
        ("weight", table ());
        ("indices", i64_tensor [ 2; 3 ] [ 0L; 2L; 2L; 4L; 1L; 2L ]);
      ]
    ~inputs:[ in_tensor "weight"; in_tensor "indices"; in_int "padding_idx" 2 ];
  [%expect {| aten and native agree |}]

let%expect_test
    "verify: a negative padding_idx and the gradient options are accepted" =
  verify_print ~target:"torch.ops.aten.embedding.default"
    ~bindings:
      [ ("weight", table ()); ("indices", i64_tensor [ 3 ] [ 1L; 2L; 4L ]) ]
    ~inputs:
      [ in_tensor "weight"; in_tensor "indices"; in_int "padding_idx" (-1) ];
  [%expect {| aten and native agree |}]

let%expect_test "dispatch: embedding builds one Embedding node" =
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.embedding.default"
    ~bindings:[ ("weight", table ()); ("indices", i64_tensor [ 2 ] [ 4L; 1L ]) ]
    ~inputs:[ in_tensor "weight"; in_tensor "indices"; in_int "padding_idx" 2 ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=5 C=3] ->[n0], t1 i64 [C=2] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] =
        embedding weight=t0 indices=t1 params={indices_rank=1 padding_idx=2}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {40, 41, 42, 10, 11, 12} |}]

let%expect_test
    "dispatch: an index outside the table is refused by the native engine" =
  dispatch_print ~target:"torch.ops.aten.embedding.default"
    ~bindings:
      [ ("weight", table ()); ("indices", i64_tensor [ 2 ] [ 4L; -1L ]) ]
    ~inputs:[ in_tensor "weight"; in_tensor "indices" ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.embedding.default"
    ~bindings:[ ("weight", table ()); ("indices", i64_tensor [ 2 ] [ 4L; 5L ]) ]
    ~inputs:[ in_tensor "weight"; in_tensor "indices" ]
    ~noutputs:1;
  [%expect
    {|
    eval error: embedding index -1 out of range [0, 5)
    eval error: embedding index 5 out of range [0, 5) |}]
