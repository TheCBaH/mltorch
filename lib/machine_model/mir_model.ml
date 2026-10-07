open Graph_ir
open Machine_ir
open Machine_interp
module Ml = Machine_lower.Mir_lower
module L = Machine_lower.Mir_layout_map

module Reason = struct
  type t = Lowering of Ml.Refusal.t | Route of string | Source of string

  let pp fmt = function
    | Lowering r -> Ml.Refusal.pp fmt r
    | Route s | Source s -> Fmt.string fmt s
end

module Route = Mir_model_route.Route
module Stage = Mir_model_route.Stage

(* The invocation's failure-site table, as a record decodes it: only a local
   or scan entry names anything. *)
let sites (inv : Loop_ir.Loop_bundle.invocation) =
  Array.map
    (fun (f : Loop_ir.Loop_failure.t) ->
      match f with
      | Loop_ir.Loop_failure.Local_out_of_range { local; _ } ->
          Mir_failure.Site_entry.Local_out_of_range local
      | Loop_ir.Loop_failure.Scan_lane_out_of_range { local; _ } ->
          Mir_failure.Site_entry.Scan_lane_out_of_range local
      | Loop_ir.Loop_failure.Scan_row_out_of_range { local; _ } ->
          Mir_failure.Site_entry.Scan_row_out_of_range local
      | Loop_ir.Loop_failure.Gather_out_of_range _
      | Loop_ir.Loop_failure.I64_division_by_zero
      | Loop_ir.Loop_failure.I64_division_overflow
      | Loop_ir.Loop_failure.I64_from_float _
      | Loop_ir.Loop_failure.Index_overflow _
      | Loop_ir.Loop_failure.Load_out_of_range _ ->
          Mir_failure.Site_entry.Other)
    (Loop_ir.Loop_js_failure.sites inv.Loop_ir.Loop_bundle.program)

module Refusal = struct
  type t = { invocation : int32; node : Node_id.t; reason : Reason.t }

  let pp fmt { invocation; node; reason } =
    Fmt.pf fmt "invocation %ld (%a): %a" invocation Node_id.pp node Reason.pp
      reason
end

(* One invocation's kernel: its program and the edge each buffer region
   binds. *)
module Kernel = struct
  type t = {
    invocation : Loop_ir.Loop_bundle.invocation;
    lowered : Ml.result;
    exec : Mir_model_route.Exec.t;
    edges : (Mir_id.Region.t * Tensor_id.t) list;
    outputs : Mir_id.Region.t list;  (** the regions it writes *)
    blocking : Mir_blocking.Decision.t option;  (** feedback's, if it ran *)
  }
end

type t = {
  bundle : Loop_ir.Loop_bundle.t;
  kernels : Kernel.t list;
  sigs : Tensor_sig.t Tensor_id.Map.t;  (** every edge a buffer binds *)
}

let planning ~group p =
  Mir_planning.make ~subject:(Ml.subject p) ~policy:"reference_f64"
    ~schedule:
      (if group = 1 then "scalar" else Fmt.str "scalar, groups of %d" group)
    ~precision:Mir_planning.Precision.F64 ~lanes:(Mir_type.Lanes.of_int 1)
    ~fma:Mir_planning.Fma.Forbidden ~capabilities:[]

(* The lowered program of an invocation under a blocking policy, and the
   decision feedback made. *)
let source ~route ~blocking ~pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  let ( let* ) = Result.bind in
  let lower ~group p =
    Result.map_error
      (fun r -> Reason.Lowering r)
      (Err.payload (Ml.program ~planning:(Some (planning ~group p)) p))
  in
  let exact () =
    Result.map_error
      (fun s -> Reason.Source s)
      (Ssa_backends.program Ssa_backends.Pipeline.Exact inv)
  in
  let blocked g =
    Result.map_error
      (fun s -> Reason.Source s)
      (Ssa_backends.blocked ~group:g inv)
  in
  match (pipeline, blocking) with
  | Ssa_backends.Pipeline.Planned _, _ ->
      Error (Reason.Source "a planned pipeline is not admitted")
  | ( Ssa_backends.Pipeline.Representation,
      (Mir_blocking.Policy.Feedback | Mir_blocking.Policy.Group _) ) ->
      Error (Reason.Source "blocking needs the exact pipeline")
  | ( (Ssa_backends.Pipeline.Exact | Ssa_backends.Pipeline.Representation),
      Mir_blocking.Policy.Unblocked ) ->
      let* p =
        Result.map_error
          (fun s -> Reason.Source s)
          (Ssa_backends.program pipeline inv)
      in
      let* l = lower ~group:1 p in
      Ok (l, None)
  | Ssa_backends.Pipeline.Exact, Mir_blocking.Policy.Group g -> (
      let* q = blocked g in
      match q with
      | Some q ->
          let* l = lower ~group:g q in
          Ok (l, None)
      | None ->
          let* p = exact () in
          let* l = lower ~group:1 p in
          Ok (l, None))
  | Ssa_backends.Pipeline.Exact, Mir_blocking.Policy.Feedback ->
      let* p = exact () in
      let* others =
        List.fold_right
          (fun g acc ->
            let* acc = acc in
            let* q = blocked g in
            Ok (match q with Some q -> (g, q) :: acc | None -> acc))
          Mir_blocking.groups (Ok [])
      in
      let lowered =
        List.map (fun (g, q) -> (g, lower ~group:g q)) ((1, p) :: others)
      in
      let decision =
        Mir_blocking.decide
          (List.map
             (fun (group, l) ->
               {
                 Mir_blocking.Candidate.group;
                 pressure =
                   (match l with
                   | Error r -> Error (Fmt.str "%a" Reason.pp r)
                   | Ok l ->
                       Mir_model_route.pressure route ~sites:(sites inv)
                         l.Ml.program);
               })
             lowered)
      in
      let* l = List.assoc decision.Mir_blocking.Decision.chosen lowered in
      Ok (l, Some decision)

let kernel ~route ~blocking ~pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  match source ~route ~blocking ~pipeline inv with
  | Error r -> Error r
  | Ok (lowered, decision) -> (
      match
        Mir_model_route.exec route ~sites:(sites inv) lowered.Ml.program
      with
      | Error s -> Error (Reason.Route s)
      | Ok exec ->
          (* the bundle's buffers positionally, by the SSA buffer each
                 one is *)
          let region (b : Loop_ir.Loop_buffer.t) =
            List.find_map
              (fun (e : L.Entry.t) ->
                if
                  Tensor_id.to_int b.Loop_ir.Loop_buffer.id
                  = (e.L.Entry.buffer.Ssa_ir.Ssa_buffer.id :> int)
                then Some e
                else None)
              lowered.Ml.layout
          in
          let bound =
            List.filter_map
              (fun (b, edge) -> Option.map (fun e -> (e, edge)) (region b))
              (List.combine
                 inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers
                 inv.Loop_ir.Loop_bundle.edges)
          in
          Ok
            {
              Kernel.invocation = inv;
              lowered;
              exec;
              edges =
                List.map
                  (fun ((e : L.Entry.t), edge) -> (e.L.Entry.region, edge))
                  bound;
              outputs =
                List.filter_map
                  (fun ((e : L.Entry.t), _) ->
                    match e.L.Entry.buffer.Ssa_ir.Ssa_buffer.role with
                    | Ssa_ir.Ssa_buffer.Output -> Some e.L.Entry.region
                    | Ssa_ir.Ssa_buffer.Input | Ssa_ir.Ssa_buffer.Scratch ->
                        None)
                  bound;
              blocking = decision;
            })

let prepare ?(route = Route.Generic) ?(blocking = Mir_blocking.Policy.Unblocked)
    ~pipeline (b : Loop_ir.Loop_bundle.t) =
  let results =
    List.mapi
      (fun k (inv : Loop_ir.Loop_bundle.invocation) ->
        Result.map_error
          (fun reason ->
            {
              Refusal.invocation = Int32.of_int k;
              node = inv.Loop_ir.Loop_bundle.node;
              reason;
            })
          (kernel ~route ~blocking ~pipeline inv))
      b.Loop_ir.Loop_bundle.invocations
  in
  match
    List.filter_map (function Error r -> Some r | Ok _ -> None) results
  with
  | _ :: _ as refusals -> Error refusals
  | [] ->
      let kernels = List.map Result.get_ok results in
      let sigs =
        List.fold_left
          (fun m (inv : Loop_ir.Loop_bundle.invocation) ->
            List.fold_left2
              (fun m (buf : Loop_ir.Loop_buffer.t) edge ->
                Tensor_id.Map.add edge buf.Loop_ir.Loop_buffer.sg m)
              m inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers
              inv.Loop_ir.Loop_bundle.edges)
          Tensor_id.Map.empty b.Loop_ir.Loop_bundle.invocations
      in
      Ok { bundle = b; kernels; sigs }

let invocations t = List.length t.kernels

let blocking t =
  List.filter_map
    (fun (k : Kernel.t) ->
      Option.map
        (fun d -> (k.Kernel.invocation.Loop_ir.Loop_bundle.node, d))
        k.Kernel.blocking)
    t.kernels

let generic t =
  List.map (fun (k : Kernel.t) -> k.Kernel.lowered.Ml.program) t.kernels

module Stop = struct
  type t =
    | At of {
        invocation : int32;
        node : Node_id.t;
        status : Mir_observation.Status.t;
      }
    | Bind of string
    | Missing of Tensor_id.t
    | Undefined_output of Tensor_id.t

  let pp fmt = function
    | At { invocation; node; status } -> (
        Fmt.pf fmt "invocation %ld (%a): " invocation Node_id.pp node;
        match status with
        | Mir_observation.Status.Failure row ->
            Fmt.pf fmt "failure %a(%a)" Mir_failure.pp
              row.Mir_observation.Row.failure
              Fmt.(list ~sep:(any ", ") Mir_const.pp)
              row.Mir_observation.Row.payload
        | s -> Mir_compare.Difference.pp_status fmt s)
    | Bind s -> Fmt.pf fmt "binding: %s" s
    | Missing id -> Fmt.pf fmt "no tensor for %a" Tensor_id.pp id
    | Undefined_output id ->
        Fmt.pf fmt "output %a has a byte no invocation wrote" Tensor_id.pp id
end

(* Each invocation's executed allocation traffic so far, on an allocated
   stage. *)
let traffic t =
  List.map
    (fun (k : Kernel.t) ->
      (k.Kernel.invocation, k.Kernel.exec.Mir_model_route.Exec.traffic ()))
    t.kernels

module Context = struct
  type model = t

  type t = {
    model : model;
    memory : Mir_memory.t;
    tensors : (Tensor_id.t, Mir_memory.Key.t) Hashtbl.t;
  }

  let ( let* ) = Result.bind

  (* The instance a tensor lives in, made at its first use. *)
  let instance cx id ~size =
    match Hashtbl.find_opt cx.tensors id with
    | Some key -> Ok key
    | None -> (
        match Mir_memory.alloc cx.memory ~size ~align:16L () with
        | None ->
            Error (Stop.Bind (Fmt.str "no address for %a" Tensor_id.pp id))
        | Some key ->
            Hashtbl.replace cx.tensors id key;
            Ok key)

  let write cx id tensor =
    let s = Mir_tensor_bytes.to_string tensor in
    let* key = instance cx id ~size:(Int64.of_int (String.length s)) in
    if
      Int64.equal
        (Mir_memory.size cx.memory key)
        (Int64.of_int (String.length s))
    then (
      Mir_memory.write_string cx.memory key ~offset:0L s;
      Ok ())
    else Error (Stop.Bind (Fmt.str "%a changed size" Tensor_id.pp id))

  let write_all cx ids lookup =
    List.fold_left
      (fun acc id ->
        let* () = acc in
        match lookup id with
        | Some tensor -> write cx id tensor
        | None -> Error (Stop.Missing id))
      (Ok ()) ids

  let create model ~constants =
    let cx =
      { model; memory = Mir_memory.create (); tensors = Hashtbl.create 64 }
    in
    let b = model.bundle in
    let* () = write_all cx b.Loop_ir.Loop_bundle.constants constants in
    (* a Region node's omitted operand: a constant-filled tensor *)
    let* () =
      List.fold_left
        (fun acc (inv : Loop_ir.Loop_bundle.invocation) ->
          List.fold_left
            (fun acc (s : Loop_ir.Loop_bundle.synthetic) ->
              let* () = acc in
              write cx s.Loop_ir.Loop_bundle.id
                (Tensor.materialize s.Loop_ir.Loop_bundle.shape (fun _ ->
                     s.Loop_ir.Loop_bundle.value)))
            acc inv.Loop_ir.Loop_bundle.synthetics)
        (Ok ()) b.Loop_ir.Loop_bundle.invocations
    in
    Ok cx

  let invoke ?fuel cx k (kernel : Kernel.t) =
    let program = Mir_verify.Generic.program kernel.Kernel.lowered.Ml.program in
    (* buffer regions are the same in every stage's program *)
    let size r =
      (Option.get (Mir_program.find_region program r)).Mir_region.size
    in
    let* shared =
      List.fold_left
        (fun acc (r, edge) ->
          let* m = acc in
          let* key = instance cx edge ~size:(size r) in
          Ok (Mir_id.Region.Map.add r key m))
        (Ok Mir_id.Region.Map.empty) kernel.Kernel.edges
    in
    (* an output starts with no defined byte *)
    List.iter
      (fun r ->
        let key = Mir_id.Region.Map.find r shared in
        Mir_memory.undefine cx.memory
          (Mir_memory.pointer cx.memory key ~lo:0L
             ~hi:(Mir_memory.size cx.memory key)))
      kernel.Kernel.outputs;
    let exec = kernel.Kernel.exec in
    let* binding =
      Result.map_error
        (fun s -> Stop.Bind s)
        (exec.Mir_model_route.Exec.instantiate cx.memory ~shared:(fun r ->
             Mir_id.Region.Map.find_opt r shared))
    in
    match
      exec.Mir_model_route.Exec.run ?fuel cx.memory binding
        ~invocation:(Int32.of_int k)
    with
    | Mir_observation.Status.Success -> Ok ()
    | status ->
        Error
          (Stop.At
             {
               invocation = Int32.of_int k;
               node = kernel.Kernel.invocation.Loop_ir.Loop_bundle.node;
               status;
             })

  let read cx id =
    match
      (Hashtbl.find_opt cx.tensors id, Tensor_id.Map.find_opt id cx.model.sigs)
    with
    | Some key, Some sg -> (
        match Err.payload (Tensor.create_of_sig sg) with
        | Error (`Quant_missing _) ->
            Error (Stop.Bind (Fmt.str "%a has no quantization" Tensor_id.pp id))
        | Ok tensor -> (
            let n = Int64.to_int (Mir_memory.size cx.memory key) in
            match
              Mir_tensor_bytes.fill tensor
                (Mir_memory.read_bytes cx.memory key ~offset:0L ~n)
            with
            | Ok () -> Ok tensor
            | Error _ -> Error (Stop.Undefined_output id)))
    | _ -> Error (Stop.Missing id)

  let run_prefix ?fuel cx ~inputs ~count =
    let b = cx.model.bundle in
    let* () = write_all cx b.Loop_ir.Loop_bundle.inputs inputs in
    let ran = List.filteri (fun k _ -> k < count) cx.model.kernels in
    let* () =
      List.fold_left
        (fun acc (k, kernel) ->
          let* () = acc in
          invoke ?fuel cx k kernel)
        (Ok ())
        (List.mapi (fun k kernel -> (k, kernel)) ran)
    in
    Ok
      (List.concat_map
         (fun (kernel : Kernel.t) ->
           List.filter_map
             (fun (r, edge) ->
               if List.mem r kernel.Kernel.outputs then Some edge else None)
             kernel.Kernel.edges)
         ran)

  let tensor = read

  let run ?fuel cx ~inputs =
    let* _ =
      run_prefix ?fuel cx ~inputs ~count:(List.length cx.model.kernels)
    in
    List.fold_right
      (fun id acc ->
        let* acc = acc in
        let* t = read cx id in
        Ok (t :: acc))
      cx.model.bundle.Loop_ir.Loop_bundle.outputs (Ok [])
end
