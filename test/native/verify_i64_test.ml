(* Verification never float-normalizes an int64. An I64 constant folded through a
   permute must verify against the exact payload, and a destination that differs
   only beyond 2^53 -- where [Int64.to_float] would make the two equal -- must
   not be proved [Identical]. *)

open Graph_ir
open Verify_fixtures

let shape = Graph_fixtures.nhwc ~h:2 ~w:2 ~c:1

let i64_tensor cells =
  let data = Bigarray.(Array1.of_array int64 c_layout cells) in
  Tensor.Tensor
    {
      shape;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let i64_const_permute () =
  build "i64_const_permute"
    Graph_builder.(
      let* w = constant ~shape ~fmt:(Payload.Fmt Payload.I64) () in
      permute Graph_fixtures.swap_hw w)

(* 2^53 + 1 is not representable as a float: it rounds to 2^53. *)
let big = 9_007_199_254_740_993L
let src_cells = [| big; 2L; 3L; 4L |]
let src = i64_tensor src_cells

let verify ~dst_payload =
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin
        (Rewrite.origin
           ~constants:[ (Tensor_id.of_int 0, src) ]
           (i64_const_permute ()))
    in
    let* (Rewrite.Step (final, map)) =
      lift_pass (Pass.run_all state [ Fold_const.pass ])
    in
    lift_verify
      (Map_verify.run map ~src:(Rewrite.snapshot state)
         ~src_constants:(Rewrite.constants state) ~dst:(Rewrite.snapshot final)
         ~dst_constants:
           (Tensor_id.Map.map dst_payload (Rewrite.constants final)))
  in
  Format.printf "@[<v 2>%a@]@." (pp_result Map_verify.Report.pp_verdicts) result

let%expect_test "an exact I64 fold is not proved by float equality" =
  verify ~dst_payload:Fun.id;
  [%expect
    {|
    {t0} -> {} identical: vacuous
    {t1} -> {t1} identical: unproved: eval: An int64 stage other than a bare Float_to_i64 cast has no grounded representation [exhaustive] |}]

(* The folded (permuted) payload with [big] replaced by 2^53: equal as floats,
   different as int64. *)
let%expect_test "a destination differing only past 2^53 is not proved identical"
    =
  verify ~dst_payload:(fun (Tensor.Tensor t as packed) ->
      let cells =
        Vec6.fold_coords t.Tensor.shape ~init:[] ~f:(fun acc c ->
            (match
               Tensor.read_i64_at6 packed (fun a -> Dim.to_int (Vec6.get c a))
             with
            | Ok v -> if Int64.equal v big then 9_007_199_254_740_992L else v
            | Error _ -> 0L)
            :: acc)
        |> List.rev |> Array.of_list
      in
      i64_tensor cells);
  [%expect
    {|
    {t0} -> {} identical: vacuous
    {t1} -> {t1} identical: unproved: eval: An int64 stage other than a bare Float_to_i64 cast has no grounded representation [exhaustive] |}]
