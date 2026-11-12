(* An arm's operand-list match that misses is a bug in the arm, not a model
   fault: [Op_bridge_error.arity_mismatch] raises [Invalid_argument] naming the
   op and the entry count instead of an anonymous [Assert_failure]. *)

let%expect_test "arity_mismatch names the op and the operand count" =
  (try
     ignore
       (Op_bridge_error.arity_mismatch ~op:"torch.ops.aten.max_pool2d.default"
          [ 1; 2; 3 ]
         : unit)
   with Invalid_argument msg -> print_endline msg);
  [%expect
    {|
    torch.ops.aten.max_pool2d.default: an operand list of 3 entries matched none of the arm's patterns
    |}]
