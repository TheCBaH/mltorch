(* The selected stage: one target's closed opcode family over virtual SSA
   values, with explicit constraints, implicit effects and condition state.
   Selected programs hold no opaque failure exit: a failure is record stores
   and a status return. The only target-neutral instruction is a logical event,
   kept so counting interpretation still observes the source's marks. *)

module Op = struct
  type 'op t = Event of Mir_event.t * int64 | Machine of 'op
end

module Terminator = struct
  type 'test t =
    | Branch of { test : 'test; then_ : Mir_edge.t; else_ : Mir_edge.t }
    | Jump of Mir_edge.t
    | Return of Mir_return.t

  let edges = function
    | Branch { then_; else_; _ } -> [ then_; else_ ]
    | Jump e -> [ e ]
    | Return _ -> []
end

(* What a target supplies to be verified, printed and selected for. *)
module type TARGET = sig
  type op
  type test

  val name : string
  val source : Mir_target.Source.t

  val op_features : op -> Mir_target.Feature.t list
  (** the features a form needs; none for the base ISA *)

  val uses : op -> Mir_value.t list
  val typing : op -> (Mir_type.t list, string) result
  val ordered : op -> bool
  val constraints : op -> Mir_target.Constraint.t list
  val abi : Mir_target.Abi.t

  val flags_view : Mir_target.View.t
  (** the condition-state register *)

  val stack_pointer : Mir_target.View.t

  val call_push : int64
  (** the bytes a call pushes for its return address (0 with a link register) *)

  val link : Mir_target.View.t option
  (** the register a call writes its return address to, if any *)

  val address_step : op -> int64 option
  (** [Some k] when the form's one result is its first operand plus [k]: the
      address arithmetic a frame realization uses for a large offset *)

  val frame_offset_ok : bytes:int64 -> int64 -> bool
  (** whether a frame access of [bytes] at this offset from its base encodes *)

  val stack_step_ok : int64 -> bool
  (** whether one stack-pointer adjustment by this many bytes encodes and keeps
      whatever alignment the target requires between adjustments *)

  val unit_bits : Mir_target.Bank.t -> int
  (** the width of a register unit of a bank *)

  val result_write : op -> Mir_target.Write.t
  (** how writing the form's register result affects the rest of its unit *)

  val clobbers : op -> Mir_target.View.t list
  (** implicit register writes beyond the results: fixed scratch, a call's
      caller-saved views *)

  val writes_flags : op -> bool
  (** whether the form changes condition state, with or without a [Flags] result
  *)

  val flags_defined : op -> int64
  (** the condition bits a [Flags] result defines; the rest are undefined *)

  val flags_read : op -> int64
  (** the bits read from a [Flags] use *)

  val test_uses : test -> Mir_value.t list
  val test_typing : test -> (unit, string) result
  val test_flags_read : test -> int64

  val pp_op :
    (Format.formatter -> Mir_value.t -> unit) -> Format.formatter -> op -> unit

  val pp_test :
    (Format.formatter -> Mir_value.t -> unit) ->
    Format.formatter ->
    test ->
    unit
end

module P = Mir_diagnostic.Problem

module Make (T : TARGET) = struct
  type program = (T.op Op.t, T.test Terminator.t) Mir_program.t

  (* A selected program: the target and the features it may use. *)
  type t = { features : Mir_target.Feature.t list; program : program }

  module Stage = struct
    type op = T.op Op.t
    type term = T.test Terminator.t

    let stage = Mir_diagnostic.Stage.Selected
    let operands = function Op.Event _ -> [] | Op.Machine o -> T.uses o
    let ordered = function Op.Event _ -> true | Op.Machine o -> T.ordered o

    let typing _ = function
      | Op.Event (_, n) ->
          if Int64.compare n 1L >= 0 then Ok []
          else
            Error
              (P.Typing
                 (Mir_typing.Error.Bad_immediate
                    Mir_typing.Immediate.Event_count))
      | Op.Machine o -> Result.map_error (fun s -> P.Target s) (T.typing o)

    let edges = Terminator.edges

    let term_values = function
      | Terminator.Branch { test; _ } -> T.test_uses test
      | Terminator.Jump _ -> []
      | Terminator.Return { Mir_return.values; _ } -> values

    let term_order = function
      | Terminator.Return { Mir_return.order; _ } -> Some order
      | Terminator.Branch _ | Terminator.Jump _ -> None

    let check_term _ ~results = function
      | Terminator.Return { Mir_return.values; _ } ->
          if
            List.equal Mir_type.equal results
              (List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) values)
          then Ok ()
          else Error P.Return_mismatch
      | Terminator.Branch _ | Terminator.Jump _ -> Ok ()
  end

  module C = Mir_check.Make (Stage)

  let is_flags (v : Mir_value.t) = Mir_type.equal v.Mir_value.ty Mir_type.Flags

  (* The target's own rules: features, constraint shapes, and local condition
     state — a [Flags] value is used only in its own block, with no
     condition-changing instruction between its definition and the use, and
     reads only bits its producer defines. *)
  let target_rules esc (sel : t) (f : (Stage.op, Stage.term) Mir_func.t) =
    List.iter
      (fun (b : (Stage.op, Stage.term) Mir_block.t) ->
        let reject ?instr s =
          Mir_check.make_reject esc Mir_diagnostic.Stage.Selected
            ~func:f.Mir_func.id ~block:b.Mir_block.id ?instr (P.Target s)
        in
        (* the live condition state: its value and the bits it defines *)
        let live = ref None in
        let read_flags ?instr (v : Mir_value.t) mask =
          match !live with
          | Some ((w : Mir_value.t), defined) when Mir_value.equal v w ->
              if not (Int64.equal (Int64.logand mask (Int64.lognot defined)) 0L)
              then
                reject ?instr
                  "reads a condition bit its producer leaves undefined"
          | _ ->
              reject ?instr "condition state used across a block or a clobber"
        in
        List.iter
          (fun (i : Stage.op Mir_instr.t) ->
            let instr = i.Mir_instr.id in
            match i.Mir_instr.op with
            | Op.Event _ -> ()
            | Op.Machine o ->
                List.iter
                  (fun ft ->
                    if not (List.mem ft sel.features) then
                      reject ~instr
                        ("needs feature " ^ Mir_target.Feature.name ft))
                  (T.op_features o);
                let uses = T.uses o in
                List.iter
                  (fun v ->
                    if is_flags v then read_flags ~instr v (T.flags_read o))
                  uses;
                let n_uses = List.length uses
                and n_res = List.length i.Mir_instr.results in
                List.iter
                  (function
                    | Mir_target.Constraint.Early_clobber k ->
                        if k < 0 || k >= n_res then
                          reject ~instr "constraint names no result"
                    | Mir_target.Constraint.Fixed_result { result; _ } ->
                        if result < 0 || result >= n_res then
                          reject ~instr "constraint names no result"
                    | Mir_target.Constraint.Fixed_use { use; _ } ->
                        if use < 0 || use >= n_uses then
                          reject ~instr "constraint names no use"
                    | Mir_target.Constraint.Tied { result; use } ->
                        if
                          result < 0 || result >= n_res || use < 0
                          || use >= n_uses
                        then reject ~instr "constraint names no operand"
                        else if
                          not
                            (Mir_type.equal
                               (List.nth i.Mir_instr.results result)
                                 .Mir_value.ty (List.nth uses use).Mir_value.ty)
                        then reject ~instr "tied operands of different types")
                  (T.constraints o);
                if T.writes_flags o then
                  live :=
                    List.find_map
                      (fun (r : Mir_value.t) ->
                        if is_flags r then Some (r, T.flags_defined o) else None)
                      i.Mir_instr.results)
          b.Mir_block.body;
        match b.Mir_block.terminator with
        | Terminator.Branch { test; _ } ->
            (match T.test_typing test with
            | Ok () -> ()
            | Error why -> reject why);
            List.iter
              (fun v ->
                if is_flags v then read_flags v (T.test_flags_read test))
              (T.test_uses test)
        | Terminator.Jump _ | Terminator.Return _ -> ())
      f.Mir_func.blocks

  type selected = t

  (* A verified selected program exists only as [verify]'s result. *)
  module Verified : sig
    type t

    val selected : t -> selected
    val verify : selected -> (t, Mir_diagnostic.t) Err.t
  end = struct
    type t = selected

    let selected t = t

    let verify (sel : selected) =
      Err.Escape.with_escape @@ fun esc ->
      let _, analyses = Err.Escape.or_throw esc (C.program sel.program) in
      List.iter (fun (f, _) -> target_rules esc sel f) analyses;
      sel
  end

  let verify = Verified.verify

  let pp_op n fmt = function
    | Op.Event (e, k) -> Fmt.pf fmt "event %s x%Ld" (Mir_event.name e) k
    | Op.Machine o -> T.pp_op (Mir_pp.Names.value n) fmt o

  let pp_term n fmt = function
    | Terminator.Branch { test; then_; else_ } ->
        Fmt.pf fmt "branch %a, %a, %a"
          (T.pp_test (Mir_pp.Names.value n))
          test (Mir_pp.pp_edge n) then_ (Mir_pp.pp_edge n) else_
    | Terminator.Jump e -> Fmt.pf fmt "jump %a" (Mir_pp.pp_edge n) e
    | Terminator.Return { Mir_return.values; order } ->
        Fmt.pf fmt "return %a; %a" (Mir_pp.Names.values n) values
          (Mir_pp.Names.value n) order

  let pp ?origins fmt (sel : t) =
    Fmt.pf fmt "@[<v>target %s [%a]@,%a@]" T.name
      Fmt.(list ~sep:(any " ") (using Mir_target.Feature.name string))
      sel.features
      (Mir_pp.pp_program_with ?origins ~op:pp_op ~term:pp_term
         ~edges:Terminator.edges)
      sel.program
end
