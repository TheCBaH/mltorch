open Graph_ir
open Machine_ir
open Machine_interp
module Ml = Machine_lower.Mir_lower
module L = Machine_lower.Mir_layout_map

module Reason = struct
  type t =
    | Attention of string
    | Lowering of Ml.Refusal.t
    | Route of string
    | Source of string

  let pp fmt = function
    | Lowering r -> Ml.Refusal.pp fmt r
    | Attention s | Route s | Source s -> Fmt.string fmt s
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

(* A constant packed for a contraction ({!Ssa_ir.Ssa_prepack}): the edge its
   packed copy binds, made from the source constant's edge once, when a
   context is created. *)
module Packed = struct
  type t = {
    id : Tensor_id.t;
    source : Tensor_id.t;
    pack : Ssa_ir.Ssa_prepack.Pack.t;
  }

  (* The packing a kernel's region needs, before the model names its edge. *)
  type pending = {
    region : Mir_id.Region.t;
    of_edge : Tensor_id.t;
    spec : Ssa_ir.Ssa_prepack.Pack.t;
  }
end

type t = {
  bundle : Loop_ir.Loop_bundle.t;
  kernels : Kernel.t list;
  packed : Packed.t list;
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
let source ?contractions ?bf16_pairs ?attention ?(constant = fun _ -> false)
    ~route ~blocking ~pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  let ( let* ) = Result.bind in
  (* the invocation's program with its body one attention, where the node is
     selected as one *)
  let binary32_attention =
    match attention with
    | Some (site : Attention_site.t) ->
        site.Attention_site.block.Attention_block.precision
        <> Attention_block.Precision.Bf16_paired
    | None -> false
  in
  (* the program with its body one [Attend] over the site's buffers *)
  let attend_site p =
    match attention with
    | None -> Ok p
    | Some (site : Attention_site.t) ->
        let* b =
          Result.map_error
            (fun u ->
              Reason.Attention (Fmt.str "%a" Attention_site.Unbound.pp u))
            (Attention_site.buffers site inv)
        in
        Result.map_error
          (fun s -> Reason.Attention s)
          (Attention_site.program site b p)
  in
  (* a binary32 attention is expanded before the plan, at its precision, so
     only a planned binary32 pipeline takes one *)
  let attend p =
    if binary32_attention then
      Error
        (Reason.Attention
           "a binary32 attention needs a planned binary32 pipeline")
    else attend_site p
  in
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
  (* a binary32 attention is the loops of its reference, built on the lowered
     program so the plan narrows them with everything else *)
  let expand_attention p =
    if not binary32_attention then Ok p
    else
      match attend_site p with
      | Ok p ->
          Ok
            (Ssa_ir.Ssa_expand_contract.program
               (Ssa_ir.Ssa_expand_attend.program p))
      | Error r -> Error (Fmt.str "%a" Reason.pp r)
  in
  match (pipeline, blocking) with
  | ( Ssa_backends.Pipeline.Planned { numerics; target },
      Mir_blocking.Policy.Unblocked ) ->
      (* the plan's own summary: its numerics, precision and target. The
         comparison oracle is the plan's, not the binary64 reference. *)
      let* plan =
        Result.map_error
          (fun s -> Reason.Source s)
          (Ssa_backends.plan ?contractions ?bf16_pairs ~rewrite:expand_attention
             ~numerics ~target inv)
      in
      let* () =
        if
          binary32_attention
          && plan.Ssa_ir.Ssa_plan.precision <> Ssa_ir.Ssa_numerics.Precision.F32
        then
          Error
            (Reason.Attention
               "a binary32 attention needs a binary32 plan; the plan computes \
                in binary64")
        else Ok ()
      in
      let* attended =
        if binary32_attention then Ok plan.Ssa_ir.Ssa_plan.program
        else attend plan.Ssa_ir.Ssa_plan.program
      in
      let program, packs = Ssa_ir.Ssa_prepack.program ~constant attended in
      let plan = { plan with Ssa_ir.Ssa_plan.program } in
      let* l =
        Result.map_error
          (fun r -> Reason.Lowering r)
          (Err.payload
             (Ml.program ~planning:(Some (Ml.summary ~target plan)) program))
      in
      Ok (l, None, packs)
  | Ssa_backends.Pipeline.Planned _, _ ->
      Error (Reason.Source "blocking needs the exact pipeline")
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
      let* p = attend p in
      let* l = lower ~group:1 p in
      Ok (l, None, [])
  | Ssa_backends.Pipeline.Exact, Mir_blocking.Policy.Group g -> (
      let* q = blocked g in
      match q with
      | Some q ->
          let* l = lower ~group:g q in
          Ok (l, None, [])
      | None ->
          let* p = exact () in
          let* l = lower ~group:1 p in
          Ok (l, None, []))
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
      Ok (l, Some decision, [])

let kernel ?contractions ?bf16_pairs ?attention ?(prepack = false) ~constants
    ~route ~blocking ~pipeline (inv : Loop_ir.Loop_bundle.invocation) =
  (* the edge an SSA buffer binds: the Loop buffer of its id, positionally *)
  let edge_of (id : Ssa_ir.Ssa_id.Buffer.t) =
    List.find_map
      (fun ((b : Loop_ir.Loop_buffer.t), edge) ->
        if Tensor_id.to_int b.Loop_ir.Loop_buffer.id = (id :> int) then
          Some edge
        else None)
      (List.combine inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers
         inv.Loop_ir.Loop_bundle.edges)
  in
  let constant id =
    prepack
    &&
    match edge_of id with
    | Some e -> Tensor_id.Set.mem e constants
    | None -> false
  in
  match
    source ?contractions ?bf16_pairs
      ?attention:
        (Option.bind attention (fun f -> f inv.Loop_ir.Loop_bundle.node))
      ~constant ~route ~blocking ~pipeline inv
  with
  | Error r -> Error r
  | Ok (lowered, decision, packs) -> (
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
            ( {
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
              },
              List.filter_map
                (fun (q : Ssa_ir.Ssa_prepack.Pack.t) ->
                  match
                    ( List.find_opt
                        (fun (e : L.Entry.t) ->
                          Ssa_ir.Ssa_id.Buffer.equal
                            e.L.Entry.buffer.Ssa_ir.Ssa_buffer.id
                            q.Ssa_ir.Ssa_prepack.Pack.target)
                        lowered.Ml.layout,
                      edge_of q.Ssa_ir.Ssa_prepack.Pack.source )
                  with
                  | Some e, Some of_edge ->
                      Some
                        { Packed.region = e.L.Entry.region; of_edge; spec = q }
                  | _ -> None)
                packs ))

let prepare ?(route = Route.Generic) ?(blocking = Mir_blocking.Policy.Unblocked)
    ?contractions ?bf16_pairs ?attention ?prepack ~pipeline
    (b : Loop_ir.Loop_bundle.t) =
  let constants = Tensor_id.Set.of_list b.Loop_ir.Loop_bundle.constants in
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
          (kernel ?contractions ?bf16_pairs ?attention ?prepack ~constants
             ~route ~blocking ~pipeline inv))
      b.Loop_ir.Loop_bundle.invocations
  in
  match
    List.filter_map (function Error r -> Some r | Ok _ -> None) results
  with
  | _ :: _ as refusals -> Error refusals
  | [] ->
      (* each packed copy an edge past every edge the bundle names, one per
         source and packing *)
      let next =
        ref
          (1
          + List.fold_left
              (fun m (inv : Loop_ir.Loop_bundle.invocation) ->
                List.fold_left
                  (fun m e -> max m (Tensor_id.to_int e))
                  m inv.Loop_ir.Loop_bundle.edges)
              (List.fold_left
                 (fun m e -> max m (Tensor_id.to_int e))
                 0
                 (b.Loop_ir.Loop_bundle.constants @ b.Loop_ir.Loop_bundle.inputs))
              b.Loop_ir.Loop_bundle.invocations)
      in
      let packed = ref [] in
      let id_of (p : Packed.pending) =
        let module P = Ssa_ir.Ssa_prepack.Pack in
        (* the same packing of the same constant, whatever buffer ids the
           kernels gave it *)
        let same (q : Packed.t) =
          Tensor_id.equal q.Packed.source p.Packed.of_edge
          && {
               q.Packed.pack with
               P.source = p.Packed.spec.P.source;
               target = p.Packed.spec.P.target;
             }
             = p.Packed.spec
        in
        match List.find_opt same !packed with
        | Some q -> q.Packed.id
        | None ->
            let id = Tensor_id.of_int !next in
            incr next;
            packed :=
              !packed
              @ [
                  { Packed.id; source = p.Packed.of_edge; pack = p.Packed.spec };
                ];
            id
      in
      let kernels =
        List.map
          (fun r ->
            let (k : Kernel.t), pending = Result.get_ok r in
            {
              k with
              Kernel.edges =
                k.Kernel.edges
                @ List.map
                    (fun (p : Packed.pending) -> (p.Packed.region, id_of p))
                    pending;
            })
          results
      in
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
      Ok { bundle = b; kernels; packed = !packed; sigs }

let invocations t = List.length t.kernels
let packed t = List.length t.packed

module Kernel_view = struct
  type t = {
    invocation : Loop_ir.Loop_bundle.invocation;
    program : Mir_verify.Generic.t;
    edges : (Mir_id.Region.t * Tensor_id.t) list;
    outputs : Mir_id.Region.t list;
  }
end

let kernel_views t =
  List.map
    (fun (k : Kernel.t) ->
      {
        Kernel_view.invocation = k.Kernel.invocation;
        program = k.Kernel.lowered.Ml.program;
        edges = k.Kernel.edges;
        outputs = k.Kernel.outputs;
      })
    t.kernels

let bundle t = t.bundle
let tensor_sig t id = Tensor_id.Map.find_opt id t.sigs

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
    (* each packed copy from its source's bytes *)
    let* () =
      List.fold_left
        (fun acc (q : Packed.t) ->
          let* () = acc in
          match constants q.Packed.source with
          | None -> Error (Stop.Missing q.Packed.source)
          | Some tensor ->
              let module P = Ssa_ir.Ssa_prepack.Pack in
              let src = Mir_tensor_bytes.to_string tensor in
              let spec = q.Packed.pack in
              let size =
                Int64.to_int (Contraction.Element.bytes spec.P.format)
              in
              let n = Int64.to_int (P.elements spec) in
              let out = Bytes.create (n * size) in
              for e = 0 to n - 1 do
                Bytes.blit_string src
                  (size * Int64.to_int (P.source_offset spec (Int64.of_int e)))
                  out (e * size) size
              done;
              let* key =
                instance cx q.Packed.id ~size:(Int64.of_int (n * size))
              in
              Mir_memory.write_string cx.memory key ~offset:0L
                (Bytes.to_string out);
              Ok ())
        (Ok ()) model.packed
    in
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
