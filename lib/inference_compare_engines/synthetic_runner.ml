open Inference_compare_schema
module Codecs = Inference_compare_codecs
module E = Codecs.Element_codec
module L = Codecs.Logical_value
module Plan = Inference_compare_plan
module H = Codecs.Payload_hasher

type config = {
  name : string;
  offsets : (string * int64) list;
  corrupt_imports : (string * int64) list;
  refuse_ops : string list;
  fail_nodes : string list;
  reuse_output_storage : bool;
  async_polls : int;
}

let default_config name =
  {
    name;
    offsets = [];
    corrupt_imports = [];
    refuse_ops = [];
    fail_nodes = [];
    reuse_output_storage = false;
    async_polls = 0;
  }

type counters = {
  mutable prepared_open : int;
  mutable values_live : int;
  mutable prepares : int;
  mutable imports : int;
  mutable exports : int;
  mutable runs : int;
  mutable polls : int;
  mutable executed_nodes : string list;
  mutable observed_inputs : (string * int64 list) list list;
}

type error =
  | Closed
  | Import_mismatch of string
  | Input_inventory of string
  | Node_failed of string
  | Refused_operator of string
  | Unsupported_graph of string
  | Value_released

let pp_error ppf = function
  | Closed -> Fmt.string ppf "prepared graph is closed"
  | Import_mismatch s -> Fmt.pf ppf "import mismatch: %s" s
  | Input_inventory s -> Fmt.pf ppf "input inventory: %s" s
  | Node_failed n -> Fmt.pf ppf "node %s failed" n
  | Refused_operator o -> Fmt.pf ppf "operator %s refused" o
  | Unsupported_graph s -> Fmt.pf ppf "unsupported graph: %s" s
  | Value_released -> Fmt.string ppf "value released"

type t =
  (module Inference_compare_core.Runner.GRAPH_RUNNER
     with type Graph.t = Graph_desc.t
      and type Error.t = error)

(* ---- the operator set ---------------------------------------------------- *)

type op =
  | Add
  | Sub
  | Mul
  | Neg
  | Id
  | Dup
  | Add_const of int64
  | Mul_const of int64
  | Assert_nonneg

let parse_op name =
  let const prefix mk =
    let n = String.length prefix in
    if String.length name > n && String.sub name 0 n = prefix then
      Option.map mk
        (Int64.of_string_opt (String.sub name n (String.length name - n)))
    else None
  in
  match name with
  | "add" -> Some Add
  | "sub" -> Some Sub
  | "mul" -> Some Mul
  | "neg" -> Some Neg
  | "id" -> Some Id
  | "dup" -> Some Dup
  | "assert_nonneg" -> Some Assert_nonneg
  | _ -> (
      match const "addc:" (fun k -> Add_const k) with
      | Some o -> Some o
      | None -> const "mulc:" (fun k -> Mul_const k))

(* operands, results *)
let arity = function
  | Add | Sub | Mul -> (2, 1)
  | Neg | Id | Add_const _ | Mul_const _ -> (1, 1)
  | Dup -> (1, 2)
  | Assert_nonneg -> (1, 0)

let apply op (args : int64 array list) : int64 array list =
  let map2 f a b = Array.mapi (fun i x -> f x b.(i)) a in
  match (op, args) with
  | Add, [ a; b ] -> [ map2 Int64.add a b ]
  | Sub, [ a; b ] -> [ map2 Int64.sub a b ]
  | Mul, [ a; b ] -> [ map2 Int64.mul a b ]
  | Neg, [ a ] -> [ Array.map Int64.neg a ]
  | Id, [ a ] -> [ Array.copy a ]
  | Dup, [ a ] -> [ Array.copy a; Array.copy a ]
  | Add_const k, [ a ] -> [ Array.map (Int64.add k) a ]
  | Mul_const k, [ a ] -> [ Array.map (Int64.mul k) a ]
  | Assert_nonneg, [ _ ] -> []
  | _ -> invalid_arg "Synthetic_runner.apply: arity (checked at preparation)"

(* ---- runner -------------------------------------------------------------- *)

let sha_hex s =
  match
    H.digest_string Inference_compare_sha256.registry
      Inference_compare_sha256.algorithm s
  with
  | Ok d -> Digest.Payload.hex d
  | Error _ -> String.make 64 '0'

let digest make s =
  match make Inference_compare_sha256.algorithm ~hex:(sha_hex s) with
  | Ok d -> d
  | Error _ -> invalid_arg "Synthetic_runner.digest"

let create (cfg : config) : t * counters =
  let counters =
    {
      prepared_open = 0;
      values_live = 0;
      prepares = 0;
      imports = 0;
      exports = 0;
      runs = 0;
      polls = 0;
      executed_nodes = [];
      observed_inputs = [];
    }
  in
  let module R = struct
    module Graph = struct
      type t = Graph_desc.t
    end

    module Error = struct
      type t = error

      let pp = pp_error
    end

    type compiled_node = { desc : Node_desc.t; op : op; name : string }

    module Prepared = struct
      type t = {
        graph : Graph_desc.t;
        order : compiled_node list;
        buffers : (string, int64 array) Hashtbl.t;  (** Reused outputs. *)
        mutable closed : bool;
      }
    end

    module Value = struct
      type t = {
        spec : Tensor_spec.t;
        data : int64 array;
        mutable released : bool;
      }
    end

    let fail e = Err.fail ~pp_error e

    let identity () : Engine_identity.t =
      let text mk s =
        match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
      in
      {
        engine = text Id.Engine.of_string cfg.name;
        route = text Id.Engine_route.of_string "synthetic";
        build_digest = digest Digest.Build.make "synthetic-runner-build";
        configuration_digest =
          digest Digest.Configuration.make
            (String.concat ";"
               (cfg.name
                :: List.map
                     (fun (k, v) -> Printf.sprintf "off:%s=%Ld" k v)
                     cfg.offsets
               @ List.map
                   (fun (k, v) -> Printf.sprintf "cor:%s=%Ld" k v)
                   cfg.corrupt_imports
               @ cfg.refuse_ops @ cfg.fail_nodes
               @ [
                   string_of_bool cfg.reuse_output_storage;
                   string_of_int cfg.async_polls;
                 ]));
        artifact_digest = None;
        environment_digest =
          digest Digest.Environment.make "synthetic-environment";
      }

    let prepare (g : Graph.t) =
      let ( let* ) = Err.bind in
      let* idx =
        match
          Plan.Graph_index.build ~limits:Inference_compare_extract.Body_limits.v
            ~side:Left g
        with
        | Ok i -> Ok i
        | Error _ -> fail (Unsupported_graph "invalid graph")
      in
      let* nodes =
        let rec go acc = function
          | [] -> Ok (List.rev acc)
          | nref :: rest -> (
              match Plan.Graph_index.node idx nref with
              | None -> fail (Unsupported_graph "unknown node")
              | Some d -> (
                  let name = Id.Operator.to_string d.operator.operation in
                  if
                    not
                      (Id.Dialect.equal d.operator.dialect
                         Synthetic_dialect.dialect_id)
                  then fail (Refused_operator name)
                  else if List.mem name cfg.refuse_ops then
                    fail (Refused_operator name)
                  else
                    match parse_op name with
                    | None -> fail (Refused_operator name)
                    | Some op ->
                        let ins, outs = arity op in
                        if
                          List.length d.operands <> ins
                          || List.length d.results <> outs
                        then fail (Unsupported_graph ("arity of " ^ name))
                        else
                          go
                            ({
                               desc = d;
                               op;
                               name = Id.Node_local.to_string d.node.local_id;
                             }
                            :: acc)
                            rest))
        in
        go [] (Plan.Graph_index.topological idx)
      in
      counters.prepares <- counters.prepares + 1;
      counters.prepared_open <- counters.prepared_open + 1;
      Ok
        {
          Prepared.graph = g;
          order = nodes;
          buffers = Hashtbl.create 8;
          closed = false;
        }

    let import ctx (p : Prepared.t) (spec : Tensor_spec.t) (v : L.t) =
      if p.closed then fail Closed
      else if not (Tensor_spec.equal spec (L.spec v)) then
        fail (Import_mismatch "spec")
      else if Id.Codec.to_string spec.dtype.codec <> "i64" then
        fail (Import_mismatch "only i64 tensors")
      else
        let codec = L.codec ctx v in
        let n = L.element_count v in
        let data =
          Array.init n (fun i -> E.integer_value codec (L.contents v) i)
        in
        counters.imports <- counters.imports + 1;
        counters.values_live <- counters.values_live + 1;
        Ok { Value.spec; data; released = false }

    let export ctx (v : Value.t) =
      if v.released then fail Value_released
      else (
        counters.exports <- counters.exports + 1;
        match L.of_integers ctx v.spec (Array.to_list v.data) with
        | Ok x -> Ok x
        | Error _ -> fail (Import_mismatch "export"))

    let release_value (v : Value.t) =
      if v.released then
        invalid_arg "Synthetic_runner.release_value: already released";
      v.released <- true;
      counters.values_live <- counters.values_live - 1

    let close (p : Prepared.t) =
      if p.closed then invalid_arg "Synthetic_runner.close: already closed";
      p.closed <- true;
      counters.prepared_open <- counters.prepared_open - 1

    let run (p : Prepared.t) ~inputs =
      let ( let* ) = Err.bind in
      if p.closed then fail Closed
      else
        let sources =
          List.map (fun (s : Graph_source.t) -> s.tensor) p.graph.sources
        in
        let* () =
          if List.length inputs <> List.length sources then
            fail (Input_inventory "wrong number of inputs")
          else if
            List.exists
              (fun s ->
                not (List.exists (fun (t, _) -> Tensor_ref.equal t s) inputs))
              sources
          then fail (Input_inventory "missing or foreign input")
          else Ok ()
        in
        let env : (Tensor_ref.t, int64 array) Hashtbl.t = Hashtbl.create 16 in
        let* () =
          let rec go = function
            | [] -> Ok ()
            | ((t : Tensor_ref.t), (v : Value.t)) :: rest ->
                if v.released then fail Value_released
                else
                  let src =
                    List.find
                      (fun (s : Graph_source.t) -> Tensor_ref.equal s.tensor t)
                      p.graph.sources
                  in
                  if not (Tensor_spec.equal src.spec v.spec) then
                    fail (Input_inventory "input spec")
                  else
                    let delta =
                      Option.value ~default:0L
                        (List.assoc_opt
                           (Id.Tensor_local.to_string t.local_id)
                           cfg.corrupt_imports)
                    in
                    Hashtbl.replace env t (Array.map (Int64.add delta) v.data);
                    go rest
          in
          go inputs
        in
        counters.runs <- counters.runs + 1;
        counters.observed_inputs <-
          List.map
            (fun (t : Tensor_ref.t) ->
              ( Id.Tensor_local.to_string t.local_id,
                Array.to_list (Hashtbl.find env t) ))
            sources
          :: counters.observed_inputs;
        let* () =
          let rec go = function
            | [] -> Ok ()
            | (n : compiled_node) :: rest ->
                counters.executed_nodes <- n.name :: counters.executed_nodes;
                if List.mem n.name cfg.fail_nodes then fail (Node_failed n.name)
                else
                  let args = List.map (Hashtbl.find env) n.desc.operands in
                  if
                    n.op = Assert_nonneg
                    && Array.exists
                         (fun x -> Int64.compare x 0L < 0)
                         (List.hd args)
                  then fail (Node_failed n.name)
                  else
                    let offset =
                      Option.value ~default:0L
                        (List.assoc_opt n.name cfg.offsets)
                    in
                    let results = apply n.op args in
                    List.iter2
                      (fun r a ->
                        Hashtbl.replace env r (Array.map (Int64.add offset) a))
                      n.desc.results results;
                    go rest
          in
          go p.order
        in
        for _ = 1 to cfg.async_polls do
          counters.polls <- counters.polls + 1
        done;
        let outputs =
          List.map
            (fun (t : Tensor_ref.t) ->
              let data = Hashtbl.find env t in
              let spec =
                match
                  List.assoc_opt t
                    (List.map (fun (r, s) -> (r, s)) p.graph.tensor_specs)
                with
                | Some s -> s
                | None -> (
                    match
                      List.find_opt
                        (fun (s : Graph_source.t) ->
                          Tensor_ref.equal s.tensor t)
                        p.graph.sources
                    with
                    | Some s -> s.spec
                    | None -> invalid_arg "Synthetic_runner.run: output spec")
              in
              let key = Id.Tensor_local.to_string t.local_id in
              let data =
                if cfg.reuse_output_storage then (
                  match Hashtbl.find_opt p.buffers key with
                  | Some buf when Array.length buf = Array.length data ->
                      Array.blit data 0 buf 0 (Array.length data);
                      buf
                  | _ ->
                      Hashtbl.replace p.buffers key data;
                      data)
                else data
              in
              counters.values_live <- counters.values_live + 1;
              (t, { Value.spec; data; released = false }))
            p.graph.final_outputs
        in
        Ok outputs
  end in
  ( (module R : Inference_compare_core.Runner.GRAPH_RUNNER
      with type Graph.t = Graph_desc.t
       and type Error.t = error),
    counters )
