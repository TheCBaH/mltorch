(* The JavaScript routes as graph engines over Native bodies. Each runs in the
   node helper under a handle; the helper process is shared by every body
   prepared through one engine and ends with the last of them. A body the
   helper's engine declines is the engine's own refusal. *)

open Inference_compare_schema
module W = Inference_compare_js_wire.Wire
module P = Inference_compare_pt2
module Tid = Graph_ir.Tensor_id

(* ---- locating the helper ---- *)

let helper_name = "js/jsoo/compare_js_engine/compare_js_engine.bc.js"

(* The helper next to the executable's build tree, unless the environment
   names one. *)
let helper_path () =
  match Sys.getenv_opt "MLTORCH_COMPARE_JS" with
  | Some p when p <> "" -> Some p
  | _ ->
      let rec up dir =
        let candidate = Filename.concat dir helper_name in
        if Sys.file_exists candidate then Some candidate
        else
          let parent = Filename.dirname dir in
          if parent = dir then None else up parent
      in
      up (Filename.dirname Sys.executable_name)

let node () = !Loop_wasm_exec.Node.node

let helper () =
  match helper_path () with
  | Some p when Sys.file_exists p -> Ok p
  | _ ->
      Error
        (Printf.sprintf
           "the JavaScript helper %s is not built (dune build \
            js/jsoo/compare_js_engine), or name it in MLTORCH_COMPARE_JS"
           helper_name)

(* ---- sessions ---- *)

type shared = { mutable session : Node_session.t option }

let acquire sh ~helper =
  match sh.session with
  | Some s when not s.Node_session.closed -> Ok s
  | _ ->
      Result.map
        (fun s ->
          sh.session <- Some s;
          s)
        (Node_session.start ~node:(node ()) ~helper)

let text_of = function
  | W.Refused m | W.Failed m -> m
  | _ -> "unexpected response"

let hello s =
  match Node_session.request s W.Hello with
  | Ok (W.Welcome { engines; runtime }) -> Ok (engines, runtime)
  | Ok r -> Error (text_of r)
  | Error m -> Error m

(* ---- the engines ---- *)

module Engine (X : sig
  val name : string
end) =
struct
  type op = Graph_ir.op
  type compiled = { session : Node_session.t; shared : shared; handle : int }

  let shared = { session = None }
  let next = ref 0

  let compile (g : Graph_ir.graph) =
    let ( let* ) = Result.bind in
    let* helper = helper () in
    let* graph = Jsont_bytesrw.encode_string Graph_ir.graph_jsont g in
    let* session = acquire shared ~helper in
    incr next;
    let handle = !next in
    match
      Node_session.request session
        (W.Prepare { handle; engine = X.name; graph })
    with
    | Ok W.Prepared ->
        session.Node_session.live <- session.live + 1;
        Ok { session; shared; handle }
    | r ->
        if session.live = 0 then (
          Node_session.close session;
          shared.session <- None);
        Error (match r with Ok r -> text_of r | Error m -> m)

  let execute c (_ : Graph_ir.graph) ~inputs =
    let inputs = List.map (fun (id, t) -> (Tid.to_int id, t)) inputs in
    match
      Node_session.request c.session (W.Run { handle = c.handle; inputs })
    with
    | Ok (W.Outputs ts) -> Ok ts
    | Ok r -> Error (text_of r)
    | Error m -> Error m

  let release c =
    ignore (Node_session.request c.session (W.Release c.handle));
    c.session.Node_session.live <- c.session.live - 1;
    if c.session.live = 0 then (
      Node_session.close c.session;
      c.shared.session <- None)
end

module Loop_js = Engine (struct
  let name = "loop-js"
end)

module Ssa_js = Engine (struct
  let name = "ssa-js"
end)

module Native_jsoo = Engine (struct
  let name = "native-jsoo"
end)

(* ---- providers ---- *)

let identity ~engine ~runtime ~helper : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string engine;
    route = id Id.Engine_route.of_string "js-node-subprocess";
    build_digest =
      P.Aten_runner.sha_digest Digest.Build.make
        (Printf.sprintf "inference-compare-js/%s/v1 protocol %d helper %s"
           engine W.version
           (Stdlib.Digest.to_hex (Stdlib.Digest.file helper)));
    configuration_digest =
      P.Aten_runner.sha_digest Digest.Configuration.make
        (String.concat ";"
           [
             "node=" ^ String.concat " " (node ());
             "engine=" ^ engine;
             "numerics=binary64 reference";
           ]);
    artifact_digest = None;
    environment_digest =
      P.Aten_runner.sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ runtime);
  }

let available () =
  let ( let* ) = Result.bind in
  let* _ = helper () in
  match Loop_c_exec.Proc.run (node () @ [ "--version" ]) with
  | Ok (Loop_c_exec.Proc.Exited 0, _) -> Ok ()
  | Ok _ -> Error "node did not report a version"
  | Error m -> Error ("node: " ^ m)

(* The helper is started once to say what it is and which engines it holds, and
   stopped: the identity is fixed before any body is prepared. *)
let instance ~engine create () =
  let ( let* ) = Result.bind in
  let* helper = helper () in
  let* s = Node_session.start ~node:(node ()) ~helper in
  let welcome = hello s in
  Node_session.close s;
  let* engines, runtime = welcome in
  if not (List.mem engine engines) then
    Error (Printf.sprintf "the helper has no engine %S" engine)
  else
    let id = identity ~engine ~runtime ~helper in
    Ok { P.Provider.runner = create ~identity:(fun () -> id); identity = id }

module Loop_js_runner = P.Graph_runner.Make (Loop_js)
module Ssa_js_runner = P.Graph_runner.Make (Ssa_js)
module Native_jsoo_runner = P.Graph_runner.Make (Native_jsoo)

let providers : P.Provider.t list =
  [
    {
      name = "loop-js";
      available;
      instantiate = instance ~engine:"loop-js" Loop_js_runner.create;
    };
    {
      name = "ssa-js";
      available;
      instantiate = instance ~engine:"ssa-js" Ssa_js_runner.create;
    };
    {
      name = "jsoo";
      available;
      instantiate = instance ~engine:"native-jsoo" Native_jsoo_runner.create;
    };
  ]

let register () = List.iter P.Provider.register providers
