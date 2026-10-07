(* The selected interpreter: the same iterative dispatcher and simultaneous
   edge transfer as the generic one, executing each target form through the
   target's own semantic function. A destructive form's tie is an allocation
   constraint only: the virtual input keeps its value. A call runs the
   callee's selected function in a frame of its own, or the helper's model;
   a helper that fails stores the model failure record into the program's
   runtime record view and returns status 1, exactly as an owned native
   helper must. *)

open Machine_ir
module D = Mir_observation.Defect

module type SEMANTICS = sig
  include Mir_sel.TARGET

  val exec : Mir_sel_env.t -> op -> Mir_datum.t list
  val test : Mir_sel_env.t -> test -> bool
end

module Make (T : SEMANTICS) = struct
  module S = Mir_sel.Make (T)

  type stop =
    | Stop_defect of D.t * Mir_interp.Location.t
    | Stop_fuel
    | Stop_unsupported of string

  type run = {
    outcome : Mir_interp.Outcome.t;
    events : (Mir_event.t * int64) list;
    steps : int64;
    last : Mir_interp.Location.t option;
  }

  let slot = function
    | Mir_event.Emitter -> 0
    | Mir_event.Key -> 1
    | Mir_event.Local -> 2
    | Mir_event.Reduction -> 3
    | Mir_event.Scan -> 4
    | Mir_event.Scan_update -> 5

  (* The runtime view a failure record is stored in: the program's one
     [Runtime] view of record size. *)
  let record_view (p : (_, _) Mir_program.t) =
    List.find_opt
      (fun (v : Mir_view.t) ->
        v.Mir_view.role = Mir_view.Runtime
        && Int64.equal v.Mir_view.size Mir_failure.record_bytes)
      p.Mir_program.views

  (* Stores a helper's failure at [p] as the model record: kind,
     invocation -1, the kind's words. *)
  let store_record env (failure : Mir_failure.t) payload
      (p : Mir_memory.Pointer.t) =
    let words =
      Mir_failure.words failure
        ~payload:(List.map (fun (c : Mir_const.t) -> c.Mir_const.bits) payload)
        ~site:None
    in
    let at k = Option.get (Mir_memory.offset_by p k) in
    Mir_sel_env.store env
      (at Mir_failure.record_kind_offset)
      ~bytes:4L ~align:4L
      (Int64.logand
         (Int64.of_int32 (Mir_failure.kind_word failure))
         0xFFFF_FFFFL);
    Mir_sel_env.store env
      (at Mir_failure.record_invocation_offset)
      ~bytes:4L ~align:4L 0xFFFF_FFFFL;
    Array.iteri
      (fun k w ->
        Mir_sel_env.store env
          (at (Mir_failure.record_word_offset k))
          ~bytes:8L ~align:8L w)
      words

  (* How a helper call stops: on a defect, or on a helper with no model. *)
  type stop_with = { defect : 'a. D.t -> 'a; unsupported : 'a. string -> 'a }

  (* A helper call through its model: results then status 0, or the record
     stored and status 1. *)
  let helper ~models ~program ~env ~stop id args =
    let defect d = stop.defect d and unsupported n = stop.unsupported n in
    match Mir_program.find_helper program id with
    | None -> defect D.Invalid_program
    | Some h -> (
        match Mir_helper_model.find models h with
        | None -> unsupported h.Mir_helper.name
        | Some m -> (
            match m.Mir_helper_model.run env.Mir_sel_env.memory args with
            | Mir_helper_model.Returns rs -> rs @ [ Mir_datum.Bits 0L ]
            | Mir_helper_model.Fails (failure, payload) ->
                if
                  not
                    (List.exists
                       (Mir_failure.equal failure)
                       h.Mir_helper.failures)
                then defect D.Invalid_program;
                (* a failing helper defines no result *)
                if h.Mir_helper.results <> [] then defect D.Invalid_program;
                (match
                   Option.bind (record_view program) (fun r ->
                       env.Mir_sel_env.view r.Mir_view.id)
                 with
                | Some p -> store_record env failure payload p
                | None -> defect D.Invalid_program);
                [ Mir_datum.Bits 1L ]))

  let run ?(fuel = 10_000_000L) ?(max_depth = 64) ?(models = [])
      (v : S.Verified.t) memory binding ~args =
    let sel = S.Verified.selected v in
    let program = sel.S.program in
    let events = Array.make 6 0L in
    let steps = ref 0L and fuel = ref fuel and last = ref None in
    let result =
      Err.Escape.with_escape @@ fun esc ->
      let view = Mir_interp.Binding.view_of binding program in
      let rec call_func ~depth (f : (S.Stage.op, S.Stage.term) Mir_func.t) args
          =
        if depth > max_depth then Err.Escape.throw esc Stop_fuel;
        let values = Hashtbl.create 64 in
        let loc =
          ref
            {
              Mir_interp.Location.func = f.Mir_func.id;
              block = f.Mir_func.entry;
              instr = None;
            }
        in
        let defect d = Err.Escape.throw esc (Stop_defect (d, !loc)) in
        let get (x : Mir_value.t) =
          match
            Hashtbl.find_opt values (Mir_id.Value.to_int x.Mir_value.id)
          with
          | Some d -> d
          | None -> defect D.Invalid_program
        in
        let set (x : Mir_value.t) d =
          Hashtbl.replace values (Mir_id.Value.to_int x.Mir_value.id) d
        in
        let rec env =
          {
            Mir_sel_env.get;
            memory;
            view;
            defect = (fun d -> defect d);
            call = (fun c a -> call ~depth c a);
          }
        and call ~depth (c : Mir_op.Callee.t) a =
          match c with
          | Mir_op.Callee.Func id -> (
              match Mir_program.find_func program id with
              | Some g -> call_func ~depth:(depth + 1) g a
              | None -> defect D.Invalid_program)
          | Mir_op.Callee.Helper id ->
              helper ~models ~program ~env
                ~stop:
                  {
                    defect = (fun d -> defect d);
                    unsupported =
                      (fun n -> Err.Escape.throw esc (Stop_unsupported n));
                  }
                id a
        in
        let tick l =
          loc := l;
          last := Some l;
          if Int64.compare !fuel 0L <= 0 then Err.Escape.throw esc Stop_fuel;
          fuel := Int64.pred !fuel;
          steps := Int64.succ !steps
        in
        let blocks = Hashtbl.create 16 in
        List.iter
          (fun (b : (_, _) Mir_block.t) ->
            Hashtbl.replace blocks (Mir_id.Block.to_int b.Mir_block.id) b)
          f.Mir_func.blocks;
        let block id = Hashtbl.find blocks (Mir_id.Block.to_int id) in
        let entry = block f.Mir_func.entry in
        if List.length entry.Mir_block.params <> List.length args then
          defect D.Invalid_program;
        List.iter2 set entry.Mir_block.params args;
        set entry.Mir_block.order Mir_datum.Order;
        let goto (e : Mir_edge.t) =
          let target = block e.Mir_edge.target in
          let vs = List.map get e.Mir_edge.args in
          List.iter2 set target.Mir_block.params vs;
          set target.Mir_block.order Mir_datum.Order;
          target
        in
        let current = ref entry and result = ref None in
        while Option.is_none !result do
          let b = !current in
          List.iter
            (fun (i : T.op Mir_sel.Op.t Mir_instr.t) ->
              tick
                {
                  Mir_interp.Location.func = f.Mir_func.id;
                  block = b.Mir_block.id;
                  instr = Some i.Mir_instr.id;
                };
              Option.iter
                (fun o -> set o.Mir_order.output Mir_datum.Order)
                i.Mir_instr.order;
              match i.Mir_instr.op with
              | Mir_sel.Op.Event (e, n) ->
                  let k = slot e in
                  events.(k) <- Int64.add events.(k) n
              | Mir_sel.Op.Machine o ->
                  let rs = T.exec env o in
                  if List.length rs <> List.length i.Mir_instr.results then
                    defect D.Invalid_program;
                  List.iter2 set i.Mir_instr.results rs
              | Mir_sel.Op.Undef v -> (
                  match env.Mir_sel_env.view v with
                  | Some p -> Mir_memory.undefine env.Mir_sel_env.memory p
                  | None -> defect D.Invalid_program))
            b.Mir_block.body;
          tick
            {
              Mir_interp.Location.func = f.Mir_func.id;
              block = b.Mir_block.id;
              instr = None;
            };
          match b.Mir_block.terminator with
          | Mir_sel.Terminator.Jump e -> current := goto e
          | Mir_sel.Terminator.Branch { test; then_; else_ } ->
              current := goto (if T.test env test then then_ else else_)
          | Mir_sel.Terminator.Return { Mir_return.values = vs; _ } ->
              result := Some (List.map get vs)
        done;
        Option.get !result
      in
      match Mir_program.find_func program program.Mir_program.main with
      | Some f -> call_func ~depth:0 f args
      | None -> invalid_arg "Mir_sel_interp: a verified program has its main"
    in
    let outcome =
      match result with
      | Ok vs -> Mir_interp.Outcome.Success vs
      | Error e -> (
          match Err.Error.kind e with
          | Stop_defect (d, l) -> Mir_interp.Outcome.Defect (d, l)
          | Stop_fuel -> Mir_interp.Outcome.Fuel_exhausted
          | Stop_unsupported s -> Mir_interp.Outcome.Unsupported s)
    in
    {
      outcome;
      events = List.map (fun e -> (e, events.(slot e))) Mir_event.all;
      steps = !steps;
      last = !last;
    }
end
