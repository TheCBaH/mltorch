(* The allocated stage: SSA destroyed, every operand a physical location.
   A block has no parameters; its entry contract lists where each selected
   parameter arrives, and every edge's transfer is explicit moves. An executed
   instruction keeps its selected form, whose virtual operand names are
   correspondence metadata for the checker only: the physical interpreter reads
   and writes the listed locations by operand position. *)

module Loc = struct
  type t =
    | Reg of Mir_target.View.t
    | Slot of { slot : Mir_id.Slot.t; bytes : int64 }
        (** a frame object before layout, read and written whole *)
    | Mem of { base : Mir_target.View.t; offset : int64; bytes : int64 }
        (** realized frame memory: [bytes] at a register's address plus [offset]
            — the stack pointer, or reserved scratch holding a large frame
            offset *)

  let equal a b =
    match (a, b) with
    | Reg a, Reg b -> Mir_target.View.equal a b
    | Slot a, Slot b ->
        Mir_id.Slot.equal a.slot b.slot && Int64.equal a.bytes b.bytes
    | Mem a, Mem b ->
        Mir_target.View.equal a.base b.base
        && Int64.equal a.offset b.offset
        && Int64.equal a.bytes b.bytes
    | (Reg _ | Slot _ | Mem _), _ -> false

  (* Whether two locations share any storage. *)
  let overlap a b =
    match (a, b) with
    | Reg a, Reg b -> Mir_target.View.overlap a b
    | Slot a, Slot b -> Mir_id.Slot.equal a.slot b.slot
    | Mem a, Mem b ->
        (* through the same base: by byte range; through different bases:
           possibly *)
        (not (Mir_target.View.equal a.base b.base))
        || Int64.compare a.offset (Int64.add b.offset b.bytes) < 0
           && Int64.compare b.offset (Int64.add a.offset a.bytes) < 0
    | (Reg _ | Slot _ | Mem _), _ -> false

  let pp fmt = function
    | Reg v -> Fmt.string fmt v.Mir_target.View.name
    | Slot { slot; bytes } -> Fmt.pf fmt "[%a:%Ld]" Mir_id.Slot.pp slot bytes
    | Mem { base; offset; bytes } ->
        Fmt.pf fmt "[%s+%Ld:%Ld]" base.Mir_target.View.name offset bytes
end

module Instr = struct
  type 'op t =
    | Exec of {
        instr : 'op Mir_sel.Op.t Mir_instr.t;
        uses : Loc.t list;  (** one per selected use, in order *)
        defs : Loc.t list;  (** one per selected result, in order *)
      }
    | Late of { op : 'op; uses : Loc.t list; defs : Loc.t list }
        (** a target form added after allocation (a large frame offset's
            address, the FP control state): no selected counterpart, its virtual
            operand names are placeholders resolved by position, and it touches
            only reserved scratch, the stack pointer and control registers *)
    | Move of {
        dst : Loc.t;
        src : Loc.t;
        value : Mir_value.t;
            (** the virtual value the move transfers: a witness the checker
                verifies, never trusts *)
      }
    | Remat of { instr : 'op Mir_sel.Op.t Mir_instr.t; defs : Loc.t list }
        (** a selected instruction run again where a spilled value of it is
            needed, instead of a reload: it must read nothing, be pure and
            total, and constrain and clobber nothing, so a second run is the
            same value. It is not the instruction's one realization. *)
    | Save of { dst : Loc.t; src : Loc.t }
        (** a callee-saved register or the link register to its save area or
            back: state the function preserves, not any virtual value *)
    | Sp of int64  (** the stack pointer moved by this many bytes *)
end

module Term = struct
  type 'test t =
    | Branch of {
        test : 'test;
        uses : Loc.t list;
        then_ : Mir_id.Block.t;
        else_ : Mir_id.Block.t;
      }
    | Jump of Mir_id.Block.t
    | Return of { values : Loc.t list }
        (** the function's results, already in their convention locations *)

  let successors = function
    | Branch { then_; else_; _ } -> [ then_; else_ ]
    | Jump b -> [ b ]
    | Return _ -> []
end

(* A selected edge: the block it leaves and its position among that block's
   successors ([then] 0, [else] 1, a jump 0). *)
module Edge_ref = struct
  type t = { from : Mir_id.Block.t; index : int }
end

module Origin = struct
  type t =
    | Block of Mir_id.Block.t  (** realizes this selected block *)
    | Edge of Edge_ref.t
        (** a split block carrying one selected edge's moves *)
end

module Block = struct
  type ('op, 'test) t = {
    id : Mir_id.Block.t;
    origin : Origin.t;
    entry : (Mir_value.t * Loc.t) list;
        (** where each selected parameter is on entry: the allocator's claim *)
    body : 'op Instr.t list;
    terminator : 'test Term.t;
  }
end

module Slot = struct
  type t = { id : Mir_id.Slot.t; bytes : int64; align : int64 }
end

module Func = struct
  type ('op, 'test) t = {
    id : Mir_id.Func.t;
    name : string;
    entry : Mir_id.Block.t;
    params : (Mir_value.t * Loc.t) list;  (** where each argument arrives *)
    results : Loc.t list;  (** where each result is returned *)
    slots : Slot.t list;
    frame : int64 option;
        (** the realized frame's bytes below the entry stack pointer; [None]
            before realization *)
    blocks : ('op, 'test) Block.t list;
  }

  let find_block t id =
    List.find_opt
      (fun (b : (_, _) Block.t) -> Mir_id.Block.equal b.Block.id id)
      t.blocks
end

module Program = struct
  type ('op, 'test) t = {
    data_model : Mir_layout.Data_model.t;
    regions : Mir_region.t list;
    views : Mir_view.t list;
    helpers : Mir_helper.t list;
    funcs : ('op, 'test) Func.t list;
    main : Mir_id.Func.t;
    features : Mir_target.Feature.t list;
  }
end
