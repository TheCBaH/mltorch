(* Noncanonical imported Bool bytes: a Bool payload built outside the
   runtime can hold any byte. The policy this pins: a nonzero byte READS as
   true everywhere (printing, float reads, casts, Kernel loads), the codec
   ENCODES the logical value (so 2 and 255 export as true), and every byte the
   runtime PRODUCES is canonical 0/1. [Tensor.equal_bits] compares logical
   reads, so raw byte identity is deliberately not part of equality. *)

open Graph_ir
open Graph_direct_fixtures

let raw_bytes = [| 0; 1; 2; 255 |]

let raw =
  let data = Bigarray.(Array1.create int8_unsigned c_layout 4) in
  Array.iteri (fun i b -> data.{i} <- b) raw_bytes;
  Tensor.Tensor
    {
      shape = s1c 4;
      payload = { Payload.fmt = Payload.Bool; quant = Payload.No_quant; data };
    }

let canonical = Tensor.materialize_bool (s1c 4) (fun c -> chan c <> 0)

let cells (Tensor.Tensor t) =
  match t.Tensor.payload.Payload.fmt with
  | Payload.Bool ->
      List.init 4 (fun i -> string_of_int t.Tensor.payload.Payload.data.{i})
      |> String.concat ","
  | _ -> "not bool"

let%expect_test "reads, equality and the codec see the logical value" =
  Format.printf "%a@." Tensor.pp raw;
  Format.printf "equal to canonical {0,1,1,1}? %b@."
    (Tensor.equal_bits raw
       (Tensor.materialize_bool (s1c 4) (fun c -> chan c <> 0)));
  Format.printf "equal to {0,1,0,1}? %b@."
    (Tensor.equal_bits raw
       (Tensor.materialize_bool (s1c 4) (fun c -> chan c land 1 = 1)));
  (match Graph_json.encode_tensor raw with
  | Ok s -> print_endline s
  | Error _ -> print_endline "encode failed");
  (match Graph_json.encode_tensor raw with
  | Ok s -> (
      match Graph_json.decode_tensor s with
      | Ok t -> Format.printf "decoded bytes: %s@." (cells t)
      | Error _ -> print_endline "decode failed")
  | Error _ -> ());
  Format.printf "canonical bytes: %s@." (cells canonical);
  [%expect
    {|
    tensor bool [C=4] {0, 1, 1, 1}
    equal to canonical {0,1,1,1}? true
    equal to {0,1,0,1}? false
    {"data":{"Array":[false,true,true,true]},"fmt":"bool","quant":null,"shape":[1,1,1,1,1,4]}
    decoded bytes: 0,1,1,1
    canonical bytes: 0,1,1,1 |}]

let%expect_test
    "Direct casts and logical-not read nonzero as true and write canonical \
     bytes" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"noncanonical" ~outputs:(fun (f, n) -> [ f; n ])
          @@
          let* x = input ~fmt:Payload.(Fmt Bool) ~shape:(s1c 4) ~name:"x" () in
          let* f = to_copy ~name:"as_float" Pointwise.To_copy.Float x in
          let* n = bitwise_not ~name:"inv" x in
          return (f, n))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ raw ]))
    in
    (* By node index: [tensor_of_name] does not know these node names. *)
    let node_output i =
      match List.nth g.Graph.nodes i with
      | { Node.outputs = id :: _; _ } -> Tensor_id.Map.find id env
      | _ -> assert false
    in
    Err.return (node_output 0, node_output 1)
  in
  Format.printf "%a@."
    (pp_result (fun fmt (f, n) ->
         Format.fprintf fmt "as_float = %a@.inv = %a (bytes %s)" Tensor.pp f
           Tensor.pp n (cells n)))
    result;
  [%expect
    {|
    as_float = tensor f32 [C=4] {0, 1, 1, 1}
    inv = tensor bool [C=4] {1, 0, 0, 0} (bytes 1,0,0,0) |}]

let%expect_test "a Kernel load of a noncanonical Bool input reads true" =
  let g =
    Graph_builder.(
      build ~name:"noncanonical_kernel" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~fmt:Payload.(Fmt Bool) ~shape:(s1c 4) ~name:"x" () in
      to_copy ~name:"out" Pointwise.To_copy.Float x)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic.run g)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> Some raw)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph.outputs) result);
  [%expect {| tensor f32 [C=4] {0, 1, 1, 1} |}]
