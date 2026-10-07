(* One plan through every route. The SSA routes are run with mark counters;
   the generic route over bytes packed here from the bound tensors, never
   through the SSA memory. Comparisons go through [Mir_compare]: outputs as
   exact binary32 cells, failures by kind, static identity and payload, and
   the logical marks, including a failure's prefix. *)

open Ssa_ir
open Machine_ir
open Machine_interp
open Machine_lower

module Route = struct
  type t = { name : string; observation : Mir_observation.t }
end

let all_events counts = List.map (fun e -> (e, counts e)) Mir_event.all

let ssa_events (c : Ssa_interp.Counters.t) =
  all_events (fun e ->
      let m =
        match e with
        | Mir_event.Emitter -> Ssa_mark.Emitter
        | Mir_event.Key -> Ssa_mark.Key
        | Mir_event.Local -> Ssa_mark.Local
        | Mir_event.Reduction -> Ssa_mark.Reduction
        | Mir_event.Scan -> Ssa_mark.Scan
        | Mir_event.Scan_update -> Ssa_mark.Scan_update
      in
      Int64.of_int (Ssa_interp.Counters.mark c m))

let f32_cell x = Mir_const.f32_bits (Int32.bits_of_float x)

(* The outputs of a tensor map, as binary32 cells in row-major order. *)
let tensor_outputs (p : Ssa_program.t) m =
  List.filter_map
    (fun (b : Ssa_buffer.t) ->
      match b.Ssa_buffer.role with
      | Ssa_buffer.Output ->
          let shape = Ssa_lower.Ssa_sig.shape b in
          let n =
            Int64.to_int (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents))
          in
          let cells = Array.make n None in
          (match
             Tensor_id.Map.find_opt
               (Tensor_id.of_int (b.Ssa_buffer.id :> int))
               m
           with
          | Some t ->
              Vec6.iter shape (fun c ->
                  cells.((Vec6.offset shape c :> int)) <-
                    Some (f32_cell (Tensor.read t c)))
          | None -> ());
          Some { Mir_observation.Output.source = Ssa_buffer.source b; cells }
      | Ssa_buffer.Input | Ssa_buffer.Scratch -> None)
    p.Ssa_program.buffers

let ssa_route name ?engine plan p ~bind =
  let counters = Ssa_interp.Counters.create () in
  let r = Ssa_lower.Ssa_exec.run ~counters ?engine plan p ~bind in
  let status, outputs =
    match Err.payload r with
    | Ok m -> (Mir_observation.Status.Success, tensor_outputs p m)
    | Error (#Ssa_interp.failure as f) ->
        (Mir_observation.Status.Failure (Mir_ssa_rows.row f), [])
    | Error e ->
        ( Mir_observation.Status.Unsupported
            (Fmt.str "%a" Ssa_lower.Ssa_exec.pp_error e),
          [] )
  in
  {
    Route.name;
    observation =
      { Mir_observation.status; outputs; events = ssa_events counters };
  }

(* The bytes of a bound binary32 input, packed little-endian in row-major
   order. *)
let pack (b : Ssa_buffer.t) t =
  let shape = Ssa_lower.Ssa_sig.shape b in
  let n =
    Int64.to_int (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents))
  in
  let bytes = Bytes.make (4 * n) '\000' in
  Vec6.iter shape (fun c ->
      Bytes.set_int32_le bytes
        (4 * (Vec6.offset shape c :> int))
        (Int32.bits_of_float (Tensor.read t c)));
  Bytes.to_string bytes

let mir_outputs memory binding (layout : Mir_layout_map.Entry.t list) =
  List.filter_map
    (fun (e : Mir_layout_map.Entry.t) ->
      let b = e.Mir_layout_map.Entry.buffer in
      match b.Ssa_buffer.role with
      | Ssa_buffer.Output ->
          let key =
            Option.get
              (Mir_interp.Binding.instance binding e.Mir_layout_map.Entry.region)
          in
          let n = Int64.to_int e.Mir_layout_map.Entry.elements in
          let size = Int64.to_int e.Mir_layout_map.Entry.elem_bytes in
          let bytes =
            Mir_memory.read_bytes memory key ~offset:0L ~n:(size * n)
          in
          let ty =
            match b.Ssa_buffer.format with
            | Ssa_format.F32 -> Mir_type.F32
            | Ssa_format.F64 -> Mir_type.F64
            | _ -> Mir_type.i64
          in
          let cells =
            Array.init n (fun i ->
                let rec word k acc =
                  if k < 0 then Some acc
                  else
                    match bytes.((size * i) + k) with
                    | None -> None
                    | Some x ->
                        word (k - 1)
                          (Int64.logor (Int64.shift_left acc 8) (Int64.of_int x))
                in
                Option.map
                  (fun bits -> { Mir_const.ty; bits })
                  (word (size - 1) 0L))
          in
          Some { Mir_observation.Output.source = Ssa_buffer.source b; cells }
      | Ssa_buffer.Input | Ssa_buffer.Scratch -> None)
    layout

let observe_run (r : Mir_interp.run) ~outputs =
  let status =
    match r.Mir_interp.outcome with
    | Mir_interp.Outcome.Success _ -> Mir_observation.Status.Success
    | Mir_interp.Outcome.Failure row -> Mir_observation.Status.Failure row
    | Mir_interp.Outcome.Defect (d, _) -> Mir_observation.Status.Defect d
    | Mir_interp.Outcome.Fuel_exhausted -> Mir_observation.Status.Fuel_exhausted
    | Mir_interp.Outcome.Unsupported s -> Mir_observation.Status.Unsupported s
  in
  {
    Mir_observation.status;
    outputs =
      (match status with
      | Mir_observation.Status.Success -> outputs ()
      | _ -> []);
    events = r.Mir_interp.events;
  }

(* The generic route: [input] gives a bound input buffer's bytes; outputs
   start zeroed, the established host contract. *)
let mir_route (lowered : Mir_lower.result) ~input =
  let program = Mir_verify.Generic.program lowered.Mir_lower.program in
  let memory = Mir_memory.create () in
  let bound region =
    List.find_map
      (fun (e : Mir_layout_map.Entry.t) ->
        if not (Mir_id.Region.equal e.Mir_layout_map.Entry.region region) then
          None
        else
          let b = e.Mir_layout_map.Entry.buffer in
          match b.Ssa_buffer.role with
          | Ssa_buffer.Input -> input b
          | Ssa_buffer.Output ->
              Some (String.make (Int64.to_int (Mir_layout_map.size e)) '\000')
          | Ssa_buffer.Scratch -> None)
      lowered.Mir_lower.layout
  in
  let observation =
    match Mir_interp.instantiate program memory ~bound with
    | Error e ->
        {
          Mir_observation.status = Mir_observation.Status.Unsupported e;
          outputs = [];
          events = [];
        }
    | Ok binding ->
        let r =
          Mir_interp.run lowered.Mir_lower.program memory binding ~args:[]
        in
        observe_run r ~outputs:(fun () ->
            mir_outputs memory binding lowered.Mir_lower.layout)
  in
  { Route.name = "generic"; observation }

let verdict ~(expected : Route.t) ~(actual : Route.t) =
  match
    Mir_compare.observations ~expected:expected.Route.observation
      ~actual:actual.Route.observation ()
  with
  | Ok () -> None
  | Error d ->
      Some
        (Fmt.str "%s vs %s: %a" expected.Route.name actual.Route.name
           Mir_compare.Difference.pp d)

let status_name (o : Mir_observation.t) =
  match o.Mir_observation.status with
  | Mir_observation.Status.Success -> "ok"
  | Mir_observation.Status.Failure r ->
      Fmt.str "%a" Mir_failure.pp r.Mir_observation.Row.failure
  | s -> Fmt.str "%a" Mir_compare.Difference.pp_status s

(* Every route of one plan; the report names the first disagreement, the
   reference's own verdict against the SSA route, and the generic status. *)
let check ?mutation plan ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error e ->
      Fmt.str "refused by source lowering: %a" Ssa_lower.Ssa_lower_plan.pp_error
        e
  | Ok p -> (
      let resolved = Ssa_plan.resolve ~numerics:Ssa_numerics.Reference_f64 p in
      let p = resolved.Ssa_plan.program in
      match
        Err.payload
          (Mir_lower.program ?mutation
             ~planning:(Some (Mir_lower.summary resolved))
             p)
      with
      | Error r -> Fmt.str "refused: %a" Mir_lower.Refusal.pp r
      | Ok lowered ->
          let reference =
            Ssa_bridge.Ssa_check.compare
              ~reference:(Kernel_eval.run_plan plan ~bind)
              ~ssa:(Ssa_lower.Ssa_exec.run plan p ~bind)
          in
          let structured = ssa_route "structured" plan p ~bind in
          let cfg =
            ssa_route "cfg" ~engine:Ssa_lower.Ssa_exec.Cfg plan p ~bind
          in
          let generic =
            mir_route lowered ~input:(fun b ->
                Option.map (pack b)
                  (bind (Tensor_id.of_int (b.Ssa_buffer.id :> int))))
          in
          let disagreements =
            List.filter_map Fun.id
              [
                verdict ~expected:structured ~actual:cfg;
                verdict ~expected:structured ~actual:generic;
              ]
          in
          Fmt.str "%s [reference: %a]%s"
            (status_name generic.Route.observation)
            Ssa_bridge.Ssa_check.pp_verdict reference
            (match disagreements with
            | [] -> ""
            | d -> " DISAGREE " ^ String.concat "; " d))

(* The bytes of input cells, little-endian, by the buffer's format. *)
let pack_cells (b : Ssa_buffer.t) (cells : Ssa_memory.cells) =
  match (b.Ssa_buffer.format, cells) with
  | Ssa_format.F32, Ssa_memory.Floats a ->
      let bytes = Bytes.create (4 * Array.length a) in
      Array.iteri
        (fun i x -> Bytes.set_int32_le bytes (4 * i) (Int32.bits_of_float x))
        a;
      Bytes.to_string bytes
  | Ssa_format.F64, Ssa_memory.Floats a ->
      let bytes = Bytes.create (8 * Array.length a) in
      Array.iteri
        (fun i x -> Bytes.set_int64_le bytes (8 * i) (Int64.bits_of_float x))
        a;
      Bytes.to_string bytes
  | Ssa_format.I64, Ssa_memory.Int64s a ->
      let bytes = Bytes.create (8 * Array.length a) in
      Array.iteri (fun i x -> Bytes.set_int64_le bytes (8 * i) x) a;
      Bytes.to_string bytes
  | _ ->
      invalid_arg "Mir_source.pack_cells: a format this harness does not pack"

let ssa_cells (b : Ssa_buffer.t) (c : Ssa_memory.cells) =
  match (b.Ssa_buffer.format, c) with
  | Ssa_format.F32, Ssa_memory.Floats a ->
      Array.map (fun x -> Some (f32_cell x)) a
  | Ssa_format.F64, Ssa_memory.Floats a ->
      Array.map (fun x -> Some (Mir_const.f64 x)) a
  | Ssa_format.I64, Ssa_memory.Int64s a ->
      Array.map (fun x -> Some (Mir_const.i64 x)) a
  | _ -> invalid_arg "Mir_source.ssa_cells: a format this harness does not read"

(* A structured SSA program built directly, its buffers given as SSA cells:
   the structured and CFG interpreters (unfused FMA never: Machine IR's is
   one rounding) against the generic route. *)
let check_program ?mutation ?(fma = Mir_planning.Fma.Forbidden)
    (p : Ssa_program.t) ~(inputs : (int * Ssa_memory.cells) list) =
  let planning =
    Mir_planning.make ~subject:(Mir_lower.subject p) ~policy:"reference_f64"
      ~schedule:"scalar" ~precision:Mir_planning.Precision.F64
      ~lanes:(Mir_type.Lanes.of_int 1) ~fma ~capabilities:[]
  in
  match
    Err.payload (Mir_lower.program ?mutation ~planning:(Some planning) p)
  with
  | Error r -> Fmt.str "refused: %a" Mir_lower.Refusal.pp r
  | Ok lowered ->
      let copy = function
        | Ssa_memory.Floats a -> Ssa_memory.Floats (Array.copy a)
        | Ssa_memory.Int64s a -> Ssa_memory.Int64s (Array.copy a)
        | Ssa_memory.Ints a -> Ssa_memory.Ints (Array.copy a)
      in
      let memory () =
        List.fold_left
          (fun m (b : Ssa_buffer.t) ->
            let cells =
              match List.assoc_opt (b.Ssa_buffer.id :> int) inputs with
              | Some c -> copy c
              | None -> Ssa_memory.zeroed b
            in
            Ssa_id.Buffer.Map.add b.Ssa_buffer.id cells m)
          Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
      in
      let outputs memory =
        List.filter_map
          (fun (b : Ssa_buffer.t) ->
            match
              (b.Ssa_buffer.role, Ssa_memory.find memory b.Ssa_buffer.id)
            with
            | Ssa_buffer.Output, Some c ->
                Some
                  {
                    Mir_observation.Output.source = Ssa_buffer.source b;
                    cells = ssa_cells b c;
                  }
            | _ -> None)
          p.Ssa_program.buffers
      in
      let route name run =
        let counters = Ssa_interp.Counters.create () in
        let memory = memory () in
        let status, outs =
          match run counters memory with
          | Ok () -> (Mir_observation.Status.Success, outputs memory)
          | Error f -> (Mir_observation.Status.Failure (Mir_ssa_rows.row f), [])
        in
        {
          Route.name;
          observation =
            {
              Mir_observation.status;
              outputs = outs;
              events = ssa_events counters;
            };
        }
      in
      let structured =
        route "structured" (fun counters memory ->
            match Err.payload (Ssa_interp.run ~counters p ~memory) with
            | Ok () -> Ok ()
            | Error (#Ssa_interp.failure as f) -> Error f
            | Error (`Invalid_program _) -> invalid_arg "an invalid program")
      in
      let generic =
        mir_route lowered ~input:(fun b ->
            Option.map (pack_cells b)
              (List.assoc_opt (b.Ssa_buffer.id :> int) inputs))
      in
      let values =
        List.concat_map
          (fun (o : Mir_observation.Output.t) ->
            Array.to_list
              (Array.map
                 (function
                   | Some c -> Fmt.str "%a" Mir_const.pp c | None -> "undef")
                 o.Mir_observation.Output.cells))
          generic.Route.observation.Mir_observation.outputs
      in
      Fmt.str "%s%s%s"
        (status_name generic.Route.observation)
        (if values = [] then "" else " [" ^ String.concat " " values ^ "]")
        (match verdict ~expected:structured ~actual:generic with
        | None -> ""
        | Some d -> " DISAGREE " ^ d)

(* One lowered case, for a later stage's harness: the generic program, the
   bytes of its bound regions, and the routes it is compared against. *)
module Case = struct
  type t = {
    lowered : Mir_lower.result;
    bound : Mir_id.Region.t -> string option;
    oracle : Route.t;  (** structured SSA *)
    generic : Route.t;
  }
end

let bound_of (lowered : Mir_lower.result) ~input region =
  List.find_map
    (fun (e : Mir_layout_map.Entry.t) ->
      if not (Mir_id.Region.equal e.Mir_layout_map.Entry.region region) then
        None
      else
        let b = e.Mir_layout_map.Entry.buffer in
        match b.Ssa_buffer.role with
        | Ssa_buffer.Input -> input b
        | Ssa_buffer.Output ->
            Some (String.make (Int64.to_int (Mir_layout_map.size e)) '\000')
        | Ssa_buffer.Scratch -> None)
    lowered.Mir_lower.layout

let case_of_plan plan ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error e ->
      Error
        (Fmt.str "refused by source lowering: %a"
           Ssa_lower.Ssa_lower_plan.pp_error e)
  | Ok p -> (
      let resolved = Ssa_plan.resolve ~numerics:Ssa_numerics.Reference_f64 p in
      let p = resolved.Ssa_plan.program in
      match
        Err.payload
          (Mir_lower.program ~planning:(Some (Mir_lower.summary resolved)) p)
      with
      | Error r -> Error (Fmt.str "refused: %a" Mir_lower.Refusal.pp r)
      | Ok lowered ->
          let input b =
            Option.map (pack b)
              (bind (Tensor_id.of_int (b.Ssa_buffer.id :> int)))
          in
          Ok
            {
              Case.lowered;
              bound = bound_of lowered ~input;
              oracle = ssa_route "structured" plan p ~bind;
              generic = mir_route lowered ~input;
            })

(* The failure row a stored model record holds, decoded through the site
   table. *)
let record_row memory key ~sites =
  let bytes =
    Mir_memory.read_bytes memory key ~offset:0L
      ~n:(Int64.to_int Mir_failure.record_bytes)
  in
  let word off n =
    let rec go k acc =
      if k < 0 then Some acc
      else
        match bytes.(off + k) with
        | None -> None
        | Some x ->
            go (k - 1) (Int64.logor (Int64.shift_left acc 8) (Int64.of_int x))
    in
    go (n - 1) 0L
  in
  let words =
    List.init Mir_failure.record_words (fun k -> word (8 + (8 * k)) 8)
  in
  match (word 0 4, word 4 4, List.for_all Option.is_some words) with
  | Some kind, Some invocation, true -> (
      let v = Array.of_list (List.map Option.get words) in
      match Mir_failure.decode ~table:sites ~kind:(Int64.to_int32 kind) ~v with
      | Ok (failure, payload, site) ->
          let payload =
            List.map2
              (fun ty bits -> { Mir_const.ty; bits })
              (Mir_failure.payload failure)
              payload
          in
          let invocation =
            if Int64.equal invocation 0xFFFF_FFFFL then None
            else Some (Int64.to_int32 invocation)
          in
          Mir_observation.Status.Failure
            { Mir_observation.Row.failure; payload; invocation; site }
      | Error Mir_failure.Decode_error.Sentinel_site ->
          Mir_observation.Status.Defect Mir_observation.Defect.Sentinel_site
      | Error _ ->
          Mir_observation.Status.Defect Mir_observation.Defect.Invalid_program)
  | _ -> Mir_observation.Status.Defect Mir_observation.Defect.Uninitialized

(* The observation of a selected-stage run: status 0 is success with the
   outputs in memory; any other status reads the failure record the program
   stored in [record]. *)
let selected_observation ~(layout : Mir_layout_map.Entry.t list) ~sites ~record
    memory binding (outcome : Mir_interp.Outcome.t) events =
  let status =
    match outcome with
    | Mir_interp.Outcome.Success vs -> (
        match List.rev vs with
        | Mir_datum.Bits 0L :: _ -> Mir_observation.Status.Success
        | Mir_datum.Bits _ :: _ -> (
            match Mir_interp.Binding.instance binding record with
            | Some key -> record_row memory key ~sites
            | None ->
                Mir_observation.Status.Defect
                  Mir_observation.Defect.Invalid_program)
        | _ ->
            Mir_observation.Status.Defect Mir_observation.Defect.Invalid_program
        )
    | Mir_interp.Outcome.Failure row -> Mir_observation.Status.Failure row
    | Mir_interp.Outcome.Defect (d, _) -> Mir_observation.Status.Defect d
    | Mir_interp.Outcome.Fuel_exhausted -> Mir_observation.Status.Fuel_exhausted
    | Mir_interp.Outcome.Unsupported s -> Mir_observation.Status.Unsupported s
  in
  {
    Mir_observation.status;
    outputs =
      (match status with
      | Mir_observation.Status.Success -> mir_outputs memory binding layout
      | _ -> []);
    events;
  }

let case_of_program (p : Ssa_program.t)
    ~(inputs : (int * Ssa_memory.cells) list)
    ?(fma = Mir_planning.Fma.Forbidden) () =
  let planning =
    Mir_planning.make ~subject:(Mir_lower.subject p) ~policy:"reference_f64"
      ~schedule:"scalar" ~precision:Mir_planning.Precision.F64
      ~lanes:(Mir_type.Lanes.of_int 1) ~fma ~capabilities:[]
  in
  match Err.payload (Mir_lower.program ~planning:(Some planning) p) with
  | Error r -> Error (Fmt.str "refused: %a" Mir_lower.Refusal.pp r)
  | Ok lowered ->
      let input b =
        Option.map (pack_cells b)
          (List.assoc_opt (b.Ssa_buffer.id :> int) inputs)
      in
      let copy = function
        | Ssa_memory.Floats a -> Ssa_memory.Floats (Array.copy a)
        | Ssa_memory.Int64s a -> Ssa_memory.Int64s (Array.copy a)
        | Ssa_memory.Ints a -> Ssa_memory.Ints (Array.copy a)
      in
      let memory =
        List.fold_left
          (fun m (b : Ssa_buffer.t) ->
            Ssa_id.Buffer.Map.add b.Ssa_buffer.id
              (match List.assoc_opt (b.Ssa_buffer.id :> int) inputs with
              | Some c -> copy c
              | None -> Ssa_memory.zeroed b)
              m)
          Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
      in
      let counters = Ssa_interp.Counters.create () in
      let status =
        match Err.payload (Ssa_interp.run ~counters p ~memory) with
        | Ok () -> Mir_observation.Status.Success
        | Error (#Ssa_interp.failure as f) ->
            Mir_observation.Status.Failure (Mir_ssa_rows.row f)
        | Error (`Invalid_program _) ->
            Mir_observation.Status.Defect Mir_observation.Defect.Invalid_program
      in
      let outputs =
        List.filter_map
          (fun (b : Ssa_buffer.t) ->
            match
              (b.Ssa_buffer.role, Ssa_memory.find memory b.Ssa_buffer.id)
            with
            | Ssa_buffer.Output, Some c ->
                Some
                  {
                    Mir_observation.Output.source = Ssa_buffer.source b;
                    cells = ssa_cells b c;
                  }
            | _ -> None)
          p.Ssa_program.buffers
      in
      Ok
        {
          Case.lowered;
          bound = bound_of lowered ~input;
          oracle =
            {
              Route.name = "structured";
              observation =
                {
                  Mir_observation.status;
                  outputs =
                    (match status with
                    | Mir_observation.Status.Success -> outputs
                    | _ -> []);
                  events = ssa_events counters;
                };
            };
          generic = mir_route lowered ~input;
        }
