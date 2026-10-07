(* The physical interpreter: register units with bit-level validity, the
   destination write rule of each form, frame slots per activation (before
   layout) or a real stack region (after), explicit moves and saves, and
   branches on located operands. Nothing virtual is consulted: an
   instruction's selected operands resolve to the locations listed for them by
   position. Every register starts undefined; a declared clobber leaves
   exactly its bits undefined (AArch64's v8-v15 keep their low half across a
   call); every function exit compares each callee-saved range that was
   defined on entry.

   Realized programs ([~realized:true]) also run on a stack region: the stack
   pointer is a register holding a pointer into it; an activation may touch
   only its own frame, between the current stack pointer and its entry value;
   a pointer spilled to the stack keeps its provenance while its bytes are
   untouched; a call needs an aligned stack; an exit needs the entry stack
   pointer back and the link register's return address intact; and a
   floating-point form runs only under the FP control state the interpreter
   models (round to nearest even, no flush to zero, no traps). *)

open Machine_ir
module D = Mir_observation.Defect
module Loc = Mir_phys.Loc

module Make (T : Mir_sel_interp.SEMANTICS) = struct
  module Sel = Mir_sel_interp.Make (T)

  type content =
    | Cond of { bits : int64; defined : int64 }
    | Data of { lo : int64; hi : int64; valid_lo : int64; valid_hi : int64 }
    | Pointer of Mir_memory.Pointer.t

  type run = {
    outcome : Mir_interp.Outcome.t;
    events : (Mir_event.t * int64) list;
    steps : int64;
  }

  let mask bits =
    if bits >= 64 then -1L
    else if bits <= 0 then 0L
    else Int64.pred (Int64.shift_left 1L bits)

  (* Bits [lo, lo + bits) of a 128-bit unit as (low word, high word) masks. *)
  let range_masks lo bits =
    let in_word base =
      let a = max lo base and b = min (lo + bits) (base + 64) in
      if a >= b then 0L else Int64.shift_left (mask (b - a)) (a - base)
    in
    (in_word 0, in_word 64)

  type regs = { units : (int, content) Hashtbl.t }

  let unit_key (v : Mir_target.View.t) =
    Mir_id.Unit.to_int v.Mir_target.View.unit

  (* Seeding for tests: a register unit's full contents. *)
  let seed regs (v : Mir_target.View.t) ~lo ~hi =
    let bits = T.unit_bits v.Mir_target.View.bank in
    Hashtbl.replace regs.units (unit_key v)
      (Data
         {
           lo;
           hi;
           valid_lo = mask (min 64 bits);
           valid_hi = (if bits > 64 then mask (bits - 64) else 0L);
         })

  let peek regs (v : Mir_target.View.t) =
    Hashtbl.find_opt regs.units (unit_key v)

  let slot_of = function
    | Mir_event.Emitter -> 0
    | Mir_event.Key -> 1
    | Mir_event.Local -> 2
    | Mir_event.Reduction -> 3
    | Mir_event.Scan -> 4
    | Mir_event.Scan_update -> 5

  let stack_bytes = 0x10_0000L

  let run ?(fuel = 10_000_000L) ?(max_depth = 64) ?(models = [])
      ?(realized = false) ?(seed = fun (_ : regs) -> ())
      (p : (T.op, T.test) Mir_phys.Program.t) memory binding ~args =
    let events = Array.make 6 0L in
    let steps = ref 0L and fuel = ref fuel in
    let regs = { units = Hashtbl.create 64 } in
    let views_program =
      {
        Mir_program.data_model = p.Mir_phys.Program.data_model;
        regions = p.Mir_phys.Program.regions;
        views = p.Mir_phys.Program.views;
        helpers = p.Mir_phys.Program.helpers;
        funcs = [];
        main = p.Mir_phys.Program.main;
        planning = None;
        revision = Mir_id.Revision.of_int 0;
      }
    in
    let view = Mir_interp.Binding.view_of binding views_program in
    (* pointers stored to memory, by synthetic address, while their bytes are
       untouched; and saved register contents, which may be undefined *)
    let pointers : (int64, Mir_memory.Pointer.t) Hashtbl.t =
      Hashtbl.create 16
    in
    let saved : (int64, content option) Hashtbl.t = Hashtbl.create 16 in
    let forget_overlapping addr bytes =
      let hit tbl width =
        Hashtbl.filter_map_inplace
          (fun a v ->
            if
              Int64.compare a (Int64.add addr bytes) < 0
              && Int64.compare addr (Int64.add a width) < 0
            then None
            else Some v)
          tbl
      in
      hit pointers 8L;
      hit saved 8L
    in
    let activations = ref 0L in
    let result =
      Err.Escape.with_escape @@ fun esc ->
      let find_func id =
        List.find_opt
          (fun (f : (_, _) Mir_phys.Func.t) ->
            Mir_id.Func.equal f.Mir_phys.Func.id id)
          p.Mir_phys.Program.funcs
      in
      (if realized then
         match Mir_memory.alloc memory ~size:stack_bytes ~align:16L () with
         | Some key ->
             let top = Mir_memory.pointer memory key ~lo:0L ~hi:stack_bytes in
             Hashtbl.replace regs.units (unit_key T.stack_pointer)
               (Pointer { top with Mir_memory.Pointer.offset = stack_bytes })
         | None -> invalid_arg "Mir_phys_interp: no room for a stack");
      (* a control register starts in the modelled state; a seed may change
         it, as a caller's own settings would *)
      if realized then
        List.iter
          (fun (v : Mir_target.View.t) ->
            if v.Mir_target.View.bank = Mir_target.Bank.Control then
              Hashtbl.replace regs.units (unit_key v)
                (Data { lo = 0L; hi = 0L; valid_lo = -1L; valid_hi = 0L }))
          T.abi.Mir_target.Abi.preserved;
      seed regs;
      (* a call that pushes its return address: the stack pointer moves down
         by [call_push] around the callee, over a fresh token *)
      let pushed run =
        if realized && Int64.compare T.call_push 0L > 0 then
          let key = unit_key T.stack_pointer in
          match Hashtbl.find_opt regs.units key with
          | Some (Pointer q) -> (
              match Mir_memory.offset_by q (Int64.neg T.call_push) with
              | Some q' ->
                  activations := Int64.succ !activations;
                  ignore
                    (Mir_memory.store memory q' ~bytes:8L ~align:1L
                       (Int64.logor 0x5241_0000_0000_0000L !activations));
                  Hashtbl.replace regs.units key (Pointer q');
                  let r = run () in
                  (match Hashtbl.find_opt regs.units key with
                  | Some (Pointer q'') -> (
                      match Mir_memory.offset_by q'' T.call_push with
                      | Some back ->
                          Hashtbl.replace regs.units key (Pointer back)
                      | None -> ())
                  | _ -> ());
                  r
              | None -> run ())
          | _ -> run ()
        else run ()
      in
      let rec call_func ~depth (f : (T.op, T.test) Mir_phys.Func.t) args =
        if depth > max_depth then Err.Escape.throw esc `Fuel;
        let slots = Hashtbl.create 64 in
        let loc =
          ref
            {
              Mir_interp.Location.func = f.Mir_phys.Func.id;
              block = f.Mir_phys.Func.entry;
              instr = None;
            }
        in
        let defect d = Err.Escape.throw esc (`Defect (d, !loc)) in
        let read_view (v : Mir_target.View.t) =
          let bits = v.Mir_target.View.bits in
          if v.Mir_target.View.lo <> 0 || bits > 64 then
            defect D.Invalid_program;
          match Hashtbl.find_opt regs.units (unit_key v) with
          | None -> defect D.Uninitialized
          | Some (Cond { bits = b; defined }) ->
              Mir_datum.Flags { bits = b; defined }
          | Some (Pointer q) ->
              if bits = 64 then Mir_datum.Ptr q else defect D.Invalid_program
          | Some (Data { lo; valid_lo; _ }) ->
              let m = mask bits in
              if Int64.equal (Int64.logand valid_lo m) m then
                Mir_datum.Bits (Int64.logand lo m)
              else defect D.Uninitialized
        in
        let write_view (v : Mir_target.View.t) rule (d : Mir_datum.t) =
          let key = unit_key v and bits = v.Mir_target.View.bits in
          let ub = T.unit_bits v.Mir_target.View.bank in
          let full = mask (min 64 ub)
          and full_hi = if ub > 64 then mask (ub - 64) else 0L in
          match d with
          | Mir_datum.Flags { bits = b; defined } ->
              Hashtbl.replace regs.units key (Cond { bits = b; defined })
          | Mir_datum.Ptr q ->
              if bits <> 64 then defect D.Invalid_program;
              Hashtbl.replace regs.units key (Pointer q)
          | Mir_datum.Order -> defect D.Invalid_program
          | Mir_datum.Bits b -> (
              let m = mask bits in
              let b = Int64.logand b m in
              match (rule : Mir_target.Write.t) with
              | Mir_target.Write.Zero_upper ->
                  Hashtbl.replace regs.units key
                    (Data
                       { lo = b; hi = 0L; valid_lo = full; valid_hi = full_hi })
              | Mir_target.Write.Undefined_upper ->
                  Hashtbl.replace regs.units key
                    (Data { lo = b; hi = 0L; valid_lo = m; valid_hi = 0L })
              | Mir_target.Write.Merge -> (
                  match Hashtbl.find_opt regs.units key with
                  | Some (Data d) ->
                      Hashtbl.replace regs.units key
                        (Data
                           {
                             d with
                             lo =
                               Int64.logor
                                 (Int64.logand d.lo (Int64.lognot m))
                                 b;
                             valid_lo = Int64.logor d.valid_lo m;
                           })
                  | Some (Pointer _ | Cond _) | None ->
                      Hashtbl.replace regs.units key
                        (Data { lo = b; hi = 0L; valid_lo = m; valid_hi = 0L }))
              )
        in
        (* exactly the clobbered bits become undefined *)
        let invalidate (v : Mir_target.View.t) =
          let key = unit_key v in
          let ml, mh =
            range_masks v.Mir_target.View.lo v.Mir_target.View.bits
          in
          match Hashtbl.find_opt regs.units key with
          | Some (Data d) ->
              Hashtbl.replace regs.units key
                (Data
                   {
                     d with
                     valid_lo = Int64.logand d.valid_lo (Int64.lognot ml);
                     valid_hi = Int64.logand d.valid_hi (Int64.lognot mh);
                   })
          | Some (Pointer _) ->
              if not (Int64.equal ml 0L) then Hashtbl.remove regs.units key
          | Some (Cond _) -> Hashtbl.remove regs.units key
          | None -> ()
        in
        let sp () =
          match Hashtbl.find_opt regs.units (unit_key T.stack_pointer) with
          | Some (Pointer q) -> q
          | _ -> defect D.Invalid_program
        in
        let sp_entry = if realized then Some (sp ()) else None in
        (* the frame memory a [Mem] location names, inside this activation's
           frame: at or above the stack pointer, below its entry value *)
        let frame_pointer (base : Mir_target.View.t) offset bytes =
          let b =
            match read_view base with
            | Mir_datum.Ptr q -> q
            | _ -> defect D.Invalid_program
          in
          let q =
            match Mir_memory.offset_by b offset with
            | Some q -> q
            | None -> defect D.Bad_access
          in
          (match sp_entry with
          | Some e ->
              let o = q.Mir_memory.Pointer.offset in
              if
                Int64.compare o (sp ()).Mir_memory.Pointer.offset < 0
                || Int64.compare (Int64.add o bytes) e.Mir_memory.Pointer.offset
                   > 0
              then defect D.Bad_access
          | None -> defect D.Invalid_program);
          q
        in
        let fault = function
          | Mir_memory.Fault.Bad_access -> defect D.Bad_access
          | Mir_memory.Fault.Uninitialized -> defect D.Uninitialized
        in
        let read (l : Loc.t) =
          match l with
          | Loc.Reg v -> read_view v
          | Loc.Slot { slot; bytes } -> (
              match Hashtbl.find_opt slots (Mir_id.Slot.to_int slot) with
              | Some (d, b) when Int64.equal b bytes -> d
              | Some _ | None -> defect D.Uninitialized)
          | Loc.Mem { base; offset; bytes } -> (
              let q = frame_pointer base offset bytes in
              let addr = Mir_memory.address memory q in
              match Hashtbl.find_opt pointers addr with
              | Some ptr when Int64.equal bytes 8L -> Mir_datum.Ptr ptr
              | _ -> (
                  match Mir_memory.load memory q ~bytes ~align:1L with
                  | Ok x -> Mir_datum.Bits x
                  | Error e -> fault e))
        in
        let write rule (l : Loc.t) d =
          match l with
          | Loc.Reg v -> write_view v rule d
          | Loc.Slot { slot; bytes } -> (
              match d with
              | Mir_datum.Flags _ | Mir_datum.Order -> defect D.Invalid_program
              | Mir_datum.Bits _ | Mir_datum.Ptr _ ->
                  Hashtbl.replace slots (Mir_id.Slot.to_int slot) (d, bytes))
          | Loc.Mem { base; offset; bytes } -> (
              let q = frame_pointer base offset bytes in
              let addr = Mir_memory.address memory q in
              forget_overlapping addr bytes;
              let bits =
                match d with
                | Mir_datum.Bits x -> x
                | Mir_datum.Ptr ptr ->
                    if not (Int64.equal bytes 8L) then defect D.Invalid_program;
                    Hashtbl.replace pointers addr ptr;
                    Mir_memory.address memory ptr
                | Mir_datum.Flags _ | Mir_datum.Order ->
                    defect D.Invalid_program
              in
              match Mir_memory.store memory q ~bytes ~align:1L bits with
              | Ok () -> ()
              | Error e -> fault e)
        in
        (* a save moves a register unit's content whole, undefined bits and
           all, through its frame word *)
        let save ~dst ~src =
          match (dst, src) with
          | Loc.Mem { base; offset; bytes }, Loc.Reg v ->
              let q = frame_pointer base offset bytes in
              let addr = Mir_memory.address memory q in
              forget_overlapping addr bytes;
              let content = Hashtbl.find_opt regs.units (unit_key v) in
              (match content with
              | Some (Data { lo; _ }) ->
                  ignore (Mir_memory.store memory q ~bytes ~align:1L lo)
              | Some (Pointer ptr) ->
                  ignore
                    (Mir_memory.store memory q ~bytes ~align:1L
                       (Mir_memory.address memory ptr))
              | Some (Cond _) | None -> ());
              Hashtbl.replace saved addr content
          | Loc.Reg v, Loc.Mem { base; offset; bytes } -> (
              let q = frame_pointer base offset bytes in
              let addr = Mir_memory.address memory q in
              match Hashtbl.find_opt saved addr with
              | Some (Some c) -> Hashtbl.replace regs.units (unit_key v) c
              | Some None -> Hashtbl.remove regs.units (unit_key v)
              | None -> (
                  match Mir_memory.load memory q ~bytes ~align:1L with
                  | Ok x ->
                      write_view v Mir_target.Write.Zero_upper
                        (Mir_datum.Bits x)
                  | Error e -> fault e))
          | _ -> defect D.Invalid_program
        in
        let tick l =
          loc := l;
          if Int64.compare !fuel 0L <= 0 then Err.Escape.throw esc `Fuel;
          fuel := Int64.pred !fuel;
          steps := Int64.succ !steps
        in
        let rec env get =
          {
            Mir_sel_env.get;
            memory;
            view;
            defect = (fun d -> defect d);
            call = call get;
          }
        and call get (c : Mir_op.Callee.t) a =
          (if realized then
             let s = Mir_memory.address memory (sp ()) in
             if
               not
                 (Int64.equal (Int64.rem s T.abi.Mir_target.Abi.stack_align) 0L)
             then defect D.Stack_alignment);
          match c with
          | Mir_op.Callee.Func id -> (
              match find_func id with
              | Some g -> pushed (fun () -> call_func ~depth:(depth + 1) g a)
              | None -> defect D.Invalid_program)
          | Mir_op.Callee.Helper id ->
              Sel.helper ~models ~program:views_program ~env:(env get)
                ~stop:
                  {
                    Sel.defect = (fun d -> defect d);
                    unsupported =
                      (fun n -> Err.Escape.throw esc (`Unsupported n));
                  }
                id a
        in
        let bind operands locs =
          if List.length operands <> List.length locs then
            defect D.Invalid_program;
          (* every operand read before any result is written *)
          let bound =
            List.map2 (fun (v : Mir_value.t) l -> (v, read l)) operands locs
          in
          fun (v : Mir_value.t) ->
            match
              List.find_opt
                (fun ((w : Mir_value.t), _) -> Mir_value.equal v w)
                bound
            with
            | Some (_, d) -> d
            | None -> defect D.Invalid_program
        in
        (* the modelled FP control state is all zero *)
        let fp_ok op =
          if List.mem Mir_target.Feature.Fp (T.op_features op) then
            List.iter
              (fun (v : Mir_target.View.t) ->
                if v.Mir_target.View.bank = Mir_target.Bank.Control then
                  match Hashtbl.find_opt regs.units (unit_key v) with
                  | Some (Data { lo; _ }) when not (Int64.equal lo 0L) ->
                      Err.Escape.throw esc (`Unsupported "FP control state")
                  | _ -> ())
              T.abi.Mir_target.Abi.preserved
        in
        let exec op uses defs =
          fp_ok op;
          let get = bind (T.uses op) uses in
          let rs = T.exec (env get) op in
          List.iter invalidate (T.clobbers op);
          if List.length rs <> List.length defs then defect D.Invalid_program;
          List.iter2
            (fun l d ->
              let rule =
                match d with
                | Mir_datum.Flags _ -> Mir_target.Write.Zero_upper
                | _ -> T.result_write op
              in
              write rule l d)
            defs rs
        in
        (* the preserved bits as they are on entry, to compare on exit *)
        let preserved =
          List.filter_map
            (fun (v : Mir_target.View.t) ->
              match Hashtbl.find_opt regs.units (unit_key v) with
              | Some c -> Some (v, c)
              | None -> None)
            T.abi.Mir_target.Abi.preserved
        in
        let same_preserved (v : Mir_target.View.t) before =
          let ml, mh =
            range_masks v.Mir_target.View.lo v.Mir_target.View.bits
          in
          match (before, Hashtbl.find_opt regs.units (unit_key v)) with
          | Data a, Some (Data b) ->
              Int64.equal
                (Int64.logand a.valid_lo ml)
                (Int64.logand b.valid_lo ml)
              && Int64.equal
                   (Int64.logand a.valid_hi mh)
                   (Int64.logand b.valid_hi mh)
              && Int64.equal
                   (Int64.logand (Int64.logand a.lo a.valid_lo) ml)
                   (Int64.logand (Int64.logand b.lo b.valid_lo) ml)
              && Int64.equal
                   (Int64.logand (Int64.logand a.hi a.valid_hi) mh)
                   (Int64.logand (Int64.logand b.hi b.valid_hi) mh)
          | before, Some after -> before = after
          | _, None -> false
        in
        (* the return address a call leaves in the link register *)
        (* a return address the call pushed: intact on exit *)
        let pushed_at =
          if realized && Int64.compare T.call_push 0L > 0 then Some (sp ())
          else None
        in
        let pushed_token =
          Option.map
            (fun q -> Mir_memory.load memory q ~bytes:8L ~align:1L)
            pushed_at
        in
        let token =
          match (realized, T.link) with
          | true, Some l ->
              activations := Int64.succ !activations;
              let t = Int64.logor 0x5245_7400_0000_0000L !activations in
              write_view l Mir_target.Write.Zero_upper (Mir_datum.Bits t);
              Some (l, t)
          | _ -> None
        in
        if List.length f.Mir_phys.Func.params <> List.length args then
          defect D.Invalid_program;
        List.iter2
          (fun (_, l) a -> write Mir_target.Write.Zero_upper l a)
          f.Mir_phys.Func.params args;
        let block id =
          match Mir_phys.Func.find_block f id with
          | Some b -> b
          | None -> defect D.Invalid_program
        in
        let current = ref (block f.Mir_phys.Func.entry) and result = ref None in
        while Option.is_none !result do
          let b = !current in
          let here instr =
            {
              Mir_interp.Location.func = f.Mir_phys.Func.id;
              block = b.Mir_phys.Block.id;
              instr;
            }
          in
          List.iter
            (fun (i : T.op Mir_phys.Instr.t) ->
              match i with
              | Mir_phys.Instr.Move { dst; src; _ } ->
                  tick (here None);
                  write Mir_target.Write.Zero_upper dst (read src)
              | Mir_phys.Instr.Save { dst; src } ->
                  tick (here None);
                  save ~dst ~src
              | Mir_phys.Instr.Sp delta -> (
                  tick (here None);
                  match Mir_memory.offset_by (sp ()) delta with
                  | Some q ->
                      Hashtbl.replace regs.units (unit_key T.stack_pointer)
                        (Pointer q)
                  | None -> defect D.Bad_access)
              | Mir_phys.Instr.Late { op; uses; defs } ->
                  tick (here None);
                  exec op uses defs
              | Mir_phys.Instr.Remat { instr; defs } -> (
                  tick (here (Some instr.Mir_instr.id));
                  match instr.Mir_instr.op with
                  | Mir_sel.Op.Machine op -> exec op [] defs
                  | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ ->
                      defect D.Invalid_program)
              | Mir_phys.Instr.Exec { instr; uses; defs } -> (
                  tick (here (Some instr.Mir_instr.id));
                  match instr.Mir_instr.op with
                  | Mir_sel.Op.Event (e, n) ->
                      let k = slot_of e in
                      events.(k) <- Int64.add events.(k) n
                  | Mir_sel.Op.Machine op -> exec op uses defs
                  | Mir_sel.Op.Undef v -> (
                      match view v with
                      | Some p -> Mir_memory.undefine memory p
                      | None -> defect D.Invalid_program)))
            b.Mir_phys.Block.body;
          tick (here None);
          match b.Mir_phys.Block.terminator with
          | Mir_phys.Term.Jump t -> current := block t
          | Mir_phys.Term.Branch { test; uses; then_; else_ } ->
              let get = bind (T.test_uses test) uses in
              current := block (if T.test (env get) test then then_ else else_)
          | Mir_phys.Term.Return { values } ->
              let rs = List.map read values in
              if not (List.for_all (fun (v, c) -> same_preserved v c) preserved)
              then defect D.Preserved_state;
              (match sp_entry with
              | Some e ->
                  if not (Mir_memory.Pointer.equal e (sp ())) then
                    defect D.Preserved_state
              | None -> ());
              (match (pushed_at, pushed_token) with
              | Some q, Some (Ok t) -> (
                  match Mir_memory.load memory q ~bytes:8L ~align:1L with
                  | Ok t' when Int64.equal t t' -> ()
                  | _ -> defect D.Return_address)
              | Some _, _ -> defect D.Return_address
              | None, _ -> ());
              (match token with
              | Some (l, t) -> (
                  match Hashtbl.find_opt regs.units (unit_key l) with
                  | Some (Data { lo; valid_lo; _ })
                    when Int64.equal valid_lo (-1L) && Int64.equal lo t ->
                      ()
                  | _ -> defect D.Return_address)
              | None -> ());
              result := Some rs
        done;
        Option.get !result
      in
      match find_func p.Mir_phys.Program.main with
      | Some f -> pushed (fun () -> call_func ~depth:0 f args)
      | None -> invalid_arg "Mir_phys_interp: no main"
    in
    let outcome =
      match result with
      | Ok vs -> Mir_interp.Outcome.Success vs
      | Error e -> (
          match Err.Error.kind e with
          | `Defect (d, l) -> Mir_interp.Outcome.Defect (d, l)
          | `Fuel -> Mir_interp.Outcome.Fuel_exhausted
          | `Unsupported n -> Mir_interp.Outcome.Unsupported n)
    in
    {
      outcome;
      events = List.map (fun e -> (e, events.(slot_of e))) Mir_event.all;
      steps = !steps;
    }
end
