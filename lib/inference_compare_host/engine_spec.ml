(* Engine configuration as it appears in an experiment, and its instantiation
   into a coordinator endpoint. A new engine is one more case here; the
   coordinator never sees the name. *)

open Inference_compare_schema
module W = Wire
module R = Inference_compare_engines.Synthetic_runner
module Core = Inference_compare_core

type t = Synthetic of R.config

let node_delta : (string * int64) W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* node = W.Obj.required o "node" W.string in
      let* delta = W.Obj.required o "delta" W.int64 in
      Ok (node, delta))

let ordinal_delta : (int * int64) W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* ordinal = W.Obj.required o "ordinal" W.nonneg_int in
      let* delta = W.Obj.required o "delta" W.int64 in
      Ok (ordinal, delta))

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "synthetic" ->
          let* name = W.Obj.required o "name" W.string in
          let* offsets = W.Obj.required o "offsets" (W.list node_delta) in
          let* corrupt_imports =
            W.Obj.required o "corrupt_imports" (W.list ordinal_delta)
          in
          let* refuse_ops = W.Obj.required o "refuse_ops" (W.list W.string) in
          let* fail_nodes = W.Obj.required o "fail_nodes" (W.list W.string) in
          let* fail_imports =
            W.Obj.required o "fail_imports" (W.list W.nonneg_int)
          in
          let* reuse_output_storage =
            W.Obj.required o "reuse_output_storage" W.bool
          in
          let* async_polls = W.Obj.required o "async_polls" W.nonneg_int in
          Ok
            (Synthetic
               {
                 R.name;
                 offsets;
                 corrupt_imports;
                 refuse_ops;
                 fail_nodes;
                 fail_imports;
                 reuse_output_storage;
                 async_polls;
               })
      | tag -> W.unknown_tag ~kind:"engine" tag)

let encode = function
  | Synthetic c ->
      W.tagged "synthetic"
        [
          W.field "name" W.enc_string c.name;
          W.field "offsets"
            (W.enc_list (fun (n, d) ->
                 W.obj
                   [
                     W.field "node" W.enc_string n;
                     W.field "delta" W.enc_int64 d;
                   ]))
            c.offsets;
          W.field "corrupt_imports"
            (W.enc_list (fun (n, d) ->
                 W.obj
                   [
                     W.field "ordinal" W.enc_int n;
                     W.field "delta" W.enc_int64 d;
                   ]))
            c.corrupt_imports;
          W.field "refuse_ops" (W.enc_list W.enc_string) c.refuse_ops;
          W.field "fail_nodes" (W.enc_list W.enc_string) c.fail_nodes;
          W.field "fail_imports" (W.enc_list W.enc_int) c.fail_imports;
          W.field "reuse_output_storage" W.enc_bool c.reuse_output_storage;
          W.field "async_polls" W.enc_int c.async_polls;
        ]

(* The endpoint, the engine's identity, and a thunk reading its live-handle
   counters (for cleanup assertions in tests and reports). *)
let instantiate (t : t) ~(source : Graph_desc.t) :
    Core.Side_exec.endpoint * Engine_identity.t =
  match t with
  | Synthetic cfg ->
      let runner, _ = R.create cfg in
      let module M = (val runner) in
      let identity = M.identity () in
      let module D =
        (val Inference_compare_engines.Synthetic_dialect.make ()
            : Inference_compare_extract.Dialect.S with type graph = Graph_desc.t
             and type body = Graph_desc.t)
      in
      ( Core.Side_exec.Endpoint
          {
            runner;
            build =
              (fun ex ->
                Inference_compare_extract.Dialect.realize (module D) source ex);
          },
        identity )
