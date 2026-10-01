(* [Eval_direct.run ~on_format_mismatch]: a result is compared with its edge's
   declared signature (shape, format, quantization) and reported when it
   differs. The hook is the census that decides whether destination passing can
   trust a declared signature. See .ai/ on the tensor arena. *)

open Graph_ir

let tensor_of_sig (sg : Tensor_sig.t) =
  let v c =
    float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3) /. 4.
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | Payload.Fmt Payload.Bool ->
      Tensor.materialize_bool sg.Tensor_sig.shape (fun c -> v c > 0.)
  | _ -> Tensor.materialize sg.Tensor_sig.shape v

let census_result (g : graph) =
  let bound kind =
    List.filter_map
      (fun id ->
        if input_kind g id = kind then
          Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
        else None)
      g.Graph.inputs
  in
  let found = ref [] in
  let result =
    Eval_direct.run
      ~on_format_mismatch:(fun m -> found := m :: !found)
      ~constants:(bound Input.Constant) g ~inputs:(bound Input.Input)
  in
  (List.rev !found, result)

let census g = fst (census_result g)

let describe (m : Eval_direct.Format_mismatch.t) =
  let (Tensor.Tensor a) = m.actual in
  let (Payload.Fmt declared) = m.declared.Tensor_sig.fmt in
  let quant = function true -> "+quant" | false -> "" in
  Fmt.str "%s %a: declared %s%s %a, actual %s%s %a" m.op Tensor_id.pp m.output
    (Payload.fmt_name declared)
    (quant (Option.is_some m.declared.Tensor_sig.quant))
    Vec6.pp_shape m.declared.Tensor_sig.shape
    (Payload.fmt_name a.Tensor.payload.Payload.fmt)
    (quant
       (match a.Tensor.payload.Payload.quant with
       | Payload.Quant _ -> true
       | Payload.No_quant -> false))
    Vec6.pp_shape a.Tensor.shape

let patch_sig id f (g : graph) =
  {
    g with
    Graph.tensors = Tensor_id.Map.update id (Option.map f) g.Graph.tensors;
  }

let out_of (g : graph) = List.hd g.Graph.outputs

let%expect_test "a consistent graph reports nothing" =
  Fmt.pr "%d@."
    (List.length (census ((List.assoc "chain" Graph_fixtures.all) ())));
  [%expect {| 0 |}]

(* Each way a declaration can lie must be seen, or a clean census means
   nothing. A destination is made from the declaration, so a lie about format or
   shape is now refused when the destination is written; a lie about
   quantization has nowhere to fail there, and the hook reports it. *)
let%expect_test "format, quantization and shape mismatches are each caught" =
  let g = (List.assoc "residual" Graph_fixtures.all) () in
  let out = out_of g in
  let show label g =
    let found, result = census_result g in
    Fmt.pr "%s: %a; %a@." label
      Fmt.(list ~sep:(any "; ") string)
      (List.map describe found)
      Fmt.(
        result ~ok:(any "ran") ~error:(fun ppf e ->
            Eval_direct.pp_error ppf (Err.Error.kind e)))
      result
  in
  show "format"
    (patch_sig out
       (fun sg -> { sg with Tensor_sig.fmt = Payload.Fmt Payload.I64 })
       g);
  show "quantization"
    (patch_sig out
       (fun sg ->
         {
           sg with
           Tensor_sig.quant = Some (Quant.per_tensor ~scale:0.5 ~zero_point:0);
         })
       g);
  show "shape"
    (patch_sig out
       (fun sg -> { sg with Tensor_sig.shape = Graph_fixtures.s1c 3 })
       g);
  [%expect
    {|
    format: ; destination format i64 cannot take float writes
    quantization: Add t3: declared f32+quant [C=4], actual f32 [C=4]; ran
    shape: ; destination shape [C=3], wanted [C=4] |}]

(* Every fixture graph: the census is empty. Two fixtures declare a Permute
   output F16 on purpose (a precision-incompatible alternate layout); the
   destination is F16 too, so what they compute is what they declare. *)
let%expect_test "census over the fixture graphs" =
  List.iter
    (fun (name, build) ->
      match List.map describe (census (build ())) with
      | [] -> ()
      | ms -> Fmt.pr "%s: %a@." name Fmt.(list ~sep:(any "; ") string) ms)
    Graph_fixtures.all;
  [%expect {| |}]
