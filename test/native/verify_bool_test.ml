(* Verification over Bool constants. [Identical] compares what a cell READS, and a
   Bool cell reads its logical value (a nonzero byte is true), so a noncanonical
   imported byte verifies against its canonical spelling -- the same policy
   [bool_noncanonical_test] pins for equality. A pass that folds a Bool constant
   must reproduce the logical value, and a fold that flips one is refuted. *)

open Graph_ir
open Verify_fixtures

let shape = Graph_fixtures.nhwc ~h:2 ~w:2 ~c:1

let bool_bytes bytes =
  let data = Bigarray.(Array1.create int8_unsigned c_layout 4) in
  Array.iteri (fun i b -> data.{i} <- b) bytes;
  Tensor.Tensor
    {
      shape;
      payload = { Payload.fmt = Payload.Bool; quant = Payload.No_quant; data };
    }

(* Bool constant -> swap H/W -> Float. The permute must keep the Bool format
   (the fold materializes against the signature), and the cast reads it. *)
let bool_const_permute () =
  build "bool_const_permute"
    Graph_builder.(
      let* w = constant ~shape ~fmt:(Payload.Fmt Payload.Bool) () in
      let* p = permute Graph_fixtures.swap_hw w in
      to_copy Pointwise.To_copy.Float p)

let raw = bool_bytes [| 0; 2; 255; 1 |]

let%expect_test "verify: a Bool fold with noncanonical bytes is proved" =
  check_with
    ~constants:[ (Tensor_id.of_int 0, raw) ]
    "bool_const_permute [fold_const]" (bool_const_permute ())
    [ Fold_const.pass ];
  [%expect
    {|
    bool_const_permute [fold_const]:
      {t0} -> {} identical: vacuous
      {t1} -> {t1} identical: proved (structural, for these constants) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]

let%expect_test "verify: a Bool fold that flipped a logical value is refuted" =
  let flip (Tensor.Tensor t as packed) =
    Tensor.materialize_bool t.Tensor.shape (fun c ->
        let v = Tensor.read packed c <> 0. in
        if Dim.to_int (Vec6.get c Axis.W) = 0 then not v else v)
  in
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin
        (Rewrite.origin
           ~constants:[ (Tensor_id.of_int 0, raw) ]
           (bool_const_permute ()))
    in
    let* (Rewrite.Step (final, map)) =
      lift_pass (Pass.run_all state [ Fold_const.pass ])
    in
    lift_verify
      (Map_verify.run map ~src:(Rewrite.snapshot state)
         ~src_constants:(Rewrite.constants state) ~dst:(Rewrite.snapshot final)
         ~dst_constants:(Tensor_id.Map.map flip (Rewrite.constants final)))
  in
  Format.printf "@[<v 2>bool_const_permute, one folded value flipped:@,%a@]@."
    (pp_result Map_verify.Report.pp_verdicts)
    result;
  [%expect
    {|
    bool_const_permute, one folded value flipped:
      {t0} -> {} identical: vacuous
      {t1} -> {t1} identical: refuted: value at (0): src.t1 vs dst.t1 under {} [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]
