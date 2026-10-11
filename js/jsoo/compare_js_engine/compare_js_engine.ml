(* The helper's handler: one function from a request body to a response body,
   exported to the driver script that owns stdin and stdout. A body is prepared
   under a handle and run any number of times; an engine that cannot take a
   body refuses it when it is prepared, and nothing runs in its place. *)

open Graph_ir
module W = Inference_compare_js_wire.Wire
module Tid = Tensor_id

type body = {
  run : (Tid.t * Tensor.packed) list -> (Tensor.packed list, string) result;
}

let bodies : (int, body) Hashtbl.t = Hashtbl.create 16
let text pp e = Format.asprintf "%a" pp e

(* The outputs in graph order; a graph output the engine did not produce is a
   failure, not a gap. *)
let in_order (g : graph) env =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | id :: rest -> (
        match Tid.Map.find_opt id env with
        | Some t -> go (t :: acc) rest
        | None -> Error (Format.asprintf "output %a was not produced" Tid.pp id)
        )
  in
  go [] g.Graph.outputs

let bundle ~kernel (g : graph) =
  match Loop_ir.Loop_bundle.build g with
  | Error e -> Error (text Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
  | Ok b -> (
      match
        Err.payload
          (Loop_bundle_exec.prepare ?kernel b ~constants:(fun _ -> None))
      with
      | Error e -> Error (text Loop_bundle_exec.pp_error e)
      | Ok p ->
          Ok
            {
              run =
                (fun inputs ->
                  match
                    Err.payload
                      (Loop_bundle_exec.run p ~bind:(fun id ->
                           List.assoc_opt id inputs))
                  with
                  | Error e -> Error (text Loop_bundle_exec.pp_error e)
                  | Ok env -> in_order g env);
            })

let native (g : graph) =
  Ok
    {
      run =
        (fun inputs ->
          match Eval_direct.run g ~inputs with
          | Error e -> Error (text Eval_direct.pp_error (Err.Error.kind e))
          | Ok env -> in_order g env);
    }

let engines = [ "loop-js"; "native-jsoo"; "ssa-js" ]

let make engine g =
  match engine with
  | "loop-js" -> bundle ~kernel:None g
  | "ssa-js" ->
      bundle
        ~kernel:
          (Some
             (fun inv ->
               Ssa_backends.js ~pipeline:Ssa_backends.Pipeline.Exact inv))
        g
  | "native-jsoo" -> native g
  | e -> Error (Printf.sprintf "no engine %S" e)

let graph_of text =
  match Jsont_bytesrw.decode_string Graph_ir.graph_jsont text with
  | Ok g -> Ok g
  | Error m -> Error ("the graph does not decode: " ^ m)

let runtime () =
  let open Js_of_ocaml in
  let node =
    Js.Unsafe.get (Js.Unsafe.get Js.Unsafe.global "process") "version"
  in
  Printf.sprintf "node %s; js_of_ocaml; ocaml %s" (Js.to_string node)
    Sys.ocaml_version

let serve (req : W.request) : W.response =
  match req with
  | W.Hello -> W.Welcome { engines; runtime = runtime () }
  | W.Prepare { handle; engine; graph } -> (
      if Hashtbl.mem bodies handle then W.Refused "handle already in use"
      else
        match Result.bind (graph_of graph) (make engine) with
        | Ok b ->
            Hashtbl.replace bodies handle b;
            W.Prepared
        | Error m -> W.Refused m)
  | W.Run { handle; inputs } -> (
      match Hashtbl.find_opt bodies handle with
      | None -> W.Failed "no such prepared body"
      | Some b -> (
          let inputs = List.map (fun (id, t) -> (Tid.of_int id, t)) inputs in
          match b.run inputs with
          | Ok outs -> W.Outputs outs
          | Error m -> W.Failed m))
  | W.Release h ->
      Hashtbl.remove bodies h;
      W.Released

let handle body =
  let response =
    match W.decode_request body with
    | Error m -> W.Failed ("malformed request: " ^ m)
    | Ok req -> (
        try serve req with e -> W.Failed ("helper: " ^ Printexc.to_string e))
  in
  W.encode_response response

let () =
  let open Js_of_ocaml in
  Js.Unsafe.set Js.Unsafe.global "mlcjHandle"
    (Js.wrap_callback (fun (u8 : Typed_array.uint8Array Js.t) ->
         Typed_array.Bytes.to_uint8Array
           (Bytes.of_string (handle (Typed_array.String.of_uint8Array u8)))))
