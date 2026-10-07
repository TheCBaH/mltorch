(* A verifier rejection: the stage, where, and what is wrong, as data. *)

module Stage = struct
  type t = Allocated | Generic | Selected

  let name = function
    | Allocated -> "allocated"
    | Generic -> "generic"
    | Selected -> "selected"
end

module Problem = struct
  type t =
    | Bad_helper of Mir_id.Helper.t
    | Bad_region of Mir_id.Region.t
    | Bad_view of Mir_id.View.t
    | Branch_condition of Mir_id.Value.t
    | Duplicate_block of Mir_id.Block.t
    | Duplicate_func of Mir_id.Func.t
    | Duplicate_helper of Mir_id.Helper.t
    | Duplicate_instr of Mir_id.Instr.t
    | Duplicate_region of Mir_id.Region.t
    | Duplicate_value of Mir_id.Value.t
    | Duplicate_view of Mir_id.View.t
    | Edge_arity of { target : Mir_id.Block.t; expected : int; found : int }
    | Edge_type of { target : Mir_id.Block.t; position : int }
    | Entry_has_predecessor of Mir_id.Block.t
    | Missing_block of Mir_id.Block.t
    | Missing_main of Mir_id.Func.t
    | Not_dominated of Mir_id.Value.t
    | Order_expected
    | Order_unexpected
    | Payload_mismatch of Mir_failure.t
    | Permission of Mir_id.View.t
    | Result_mismatch
    | Return_mismatch
    | Shift_count
    | Stale_order of Mir_id.Value.t
    | Target of string
        (** a target's own constraint, named by the target's closed vocabulary
        *)
    | Typing of Mir_typing.Error.t
    | Undefined_value of Mir_id.Value.t
    | Unproven_domain
    | Unreachable_block of Mir_id.Block.t
    | Value_type of Mir_id.Value.t
        (** a block parameter, order state or result whose type cannot hold that
            role *)

  let pp fmt = function
    | Bad_helper h -> Fmt.pf fmt "malformed helper %a" Mir_id.Helper.pp h
    | Bad_region r -> Fmt.pf fmt "malformed region %a" Mir_id.Region.pp r
    | Bad_view v -> Fmt.pf fmt "malformed view %a" Mir_id.View.pp v
    | Branch_condition v ->
        Fmt.pf fmt "branch condition %a is not a predicate" Mir_id.Value.pp v
    | Duplicate_block b -> Fmt.pf fmt "duplicate block %a" Mir_id.Block.pp b
    | Duplicate_func f -> Fmt.pf fmt "duplicate function %a" Mir_id.Func.pp f
    | Duplicate_helper h -> Fmt.pf fmt "duplicate helper %a" Mir_id.Helper.pp h
    | Duplicate_instr i ->
        Fmt.pf fmt "duplicate instruction %a" Mir_id.Instr.pp i
    | Duplicate_region r -> Fmt.pf fmt "duplicate region %a" Mir_id.Region.pp r
    | Duplicate_value v -> Fmt.pf fmt "%a is defined twice" Mir_id.Value.pp v
    | Duplicate_view v -> Fmt.pf fmt "duplicate view %a" Mir_id.View.pp v
    | Edge_arity { target; expected; found } ->
        Fmt.pf fmt "edge to %a passes %d arguments for %d parameters"
          Mir_id.Block.pp target found expected
    | Edge_type { target; position } ->
        Fmt.pf fmt "edge to %a: argument %d has the wrong type" Mir_id.Block.pp
          target position
    | Entry_has_predecessor b ->
        Fmt.pf fmt "entry %a is a branch target" Mir_id.Block.pp b
    | Missing_block b -> Fmt.pf fmt "no block %a" Mir_id.Block.pp b
    | Missing_main f -> Fmt.pf fmt "no main function %a" Mir_id.Func.pp f
    | Not_dominated v ->
        Fmt.pf fmt "%a is used where its definition does not dominate"
          Mir_id.Value.pp v
    | Order_expected ->
        Fmt.string fmt "an ordered operation without order state"
    | Order_unexpected ->
        Fmt.string fmt "an unordered operation with order state"
    | Payload_mismatch f ->
        Fmt.pf fmt "failure %a: payload does not match its schema"
          Mir_failure.pp f
    | Permission v -> Fmt.pf fmt "an access %a does not permit" Mir_id.View.pp v
    | Result_mismatch -> Fmt.string fmt "results do not match the opcode"
    | Return_mismatch ->
        Fmt.string fmt "return values do not match the function"
    | Shift_count ->
        Fmt.string fmt "shift count is not a constant below the width"
    | Stale_order v -> Fmt.pf fmt "stale order state %a" Mir_id.Value.pp v
    | Target s -> Fmt.pf fmt "target constraint: %s" s
    | Typing e -> Mir_typing.Error.pp fmt e
    | Undefined_value v -> Fmt.pf fmt "%a is never defined" Mir_id.Value.pp v
    | Unproven_domain ->
        Fmt.string fmt "a domain-restricted operation without guard evidence"
    | Unreachable_block b -> Fmt.pf fmt "%a is unreachable" Mir_id.Block.pp b
    | Value_type v ->
        Fmt.pf fmt "%a has a type its role forbids" Mir_id.Value.pp v
end

type t = {
  stage : Stage.t;
  func : Mir_id.Func.t option;
  block : Mir_id.Block.t option;
  instr : Mir_id.Instr.t option;
  problem : Problem.t;
}

let pp fmt t =
  let opt pp fmt = function None -> () | Some x -> Fmt.pf fmt " %a" pp x in
  Fmt.pf fmt "%s%a%a%a: %a" (Stage.name t.stage) (opt Mir_id.Func.pp) t.func
    (opt Mir_id.Block.pp) t.block (opt Mir_id.Instr.pp) t.instr Problem.pp
    t.problem
