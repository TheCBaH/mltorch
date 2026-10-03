module Reason = struct
  type t =
    | Alignment_too_large of { align : int; natural : int }
    | Bad_label of int
    | Bad_limits of { min_pages : int; max_pages : int option }
    | Const_expected
    | Data_out_of_memory of { offset : int; length : int }
    | Duplicate_export of string
    | Global_immutable of int
    | Memory_required
    | Offset_out_of_range of int
    | Operand_mismatch of {
        expected : Wasm_type.t;
        actual : Wasm_type.t option;
      }
    | Select_operands_differ of Wasm_type.t * Wasm_type.t
    | Stack_height_mismatch of { expected : int; actual : int }
    | Stack_underflow
    | Unknown_export_target of int
    | Unknown_func of int
    | Unknown_global of int
    | Unknown_local of int

  let pp ppf = function
    | Alignment_too_large { align; natural } ->
        Fmt.pf ppf "alignment 2^%d exceeds the natural 2^%d" align natural
    | Bad_label n -> Fmt.pf ppf "branch depth %d names no enclosing block" n
    | Bad_limits { min_pages; max_pages } ->
        Fmt.pf ppf "memory limits min=%d max=%a" min_pages
          Fmt.(option ~none:(any "none") int)
          max_pages
    | Const_expected -> Fmt.string ppf "initializer is not a constant"
    | Data_out_of_memory { offset; length } ->
        Fmt.pf ppf "data segment [%d, +%d) exceeds the initial memory" offset
          length
    | Duplicate_export n -> Fmt.pf ppf "duplicate export %S" n
    | Global_immutable n -> Fmt.pf ppf "global %d is immutable" n
    | Memory_required -> Fmt.string ppf "the module has no memory"
    | Offset_out_of_range n -> Fmt.pf ppf "offset %d is outside [0, 2^31)" n
    | Operand_mismatch { expected; actual } ->
        Fmt.pf ppf "expected %a, found %a" Wasm_type.pp expected
          Fmt.(option ~none:(any "unreachable") Wasm_type.pp)
          actual
    | Select_operands_differ (a, b) ->
        Fmt.pf ppf "select operands %a and %a differ" Wasm_type.pp a
          Wasm_type.pp b
    | Stack_height_mismatch { expected; actual } ->
        Fmt.pf ppf "block ends with %d values, expected %d" actual expected
    | Stack_underflow -> Fmt.string ppf "operand stack underflow"
    | Unknown_export_target n -> Fmt.pf ppf "export of unknown index %d" n
    | Unknown_func n -> Fmt.pf ppf "unknown function %d" n
    | Unknown_global n -> Fmt.pf ppf "unknown global %d" n
    | Unknown_local n -> Fmt.pf ppf "unknown local %d" n
end

module Invalid = struct
  type t = { func : int option; reason : Reason.t }
end

type error = [ `Wasm_invalid of Invalid.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Wasm_invalid { Invalid.func; reason } ->
      Fmt.pf ppf "invalid module%a: %a"
        Fmt.(option (any " (function " ++ int ++ any ")"))
        func Reason.pp reason

(* The immediate bound on offsets, data addresses and constants of the policy:
   a positive [int32], so no check or encoding here depends on a 63-bit [int]. *)
let max_offset = 0x7FFF_FFFF

type frame = {
  label : Wasm_type.t list;  (** what a branch to this frame carries *)
  results : Wasm_type.t list;
  height : int;
  mutable unreachable : bool;
}

type state = {
  mutable vals : Wasm_type.t option list;  (** top first *)
  mutable height : int;
  mutable frames : frame list;  (** innermost first *)
}

type ctx = {
  func_types : Wasm.Func_type.t array;
  globals : Wasm.Global.t array;
  has_memory : bool;
  locals : Wasm_type.t array;
  esc : error Err.Escape.t;
  func : int option;
}

let throw ctx reason =
  Err.Escape.throw ctx.esc (`Wasm_invalid { Invalid.func = ctx.func; reason })

let push st v =
  st.vals <- v :: st.vals;
  st.height <- st.height + 1

let top_frame st =
  match st.frames with f :: _ -> f | [] -> invalid_arg "Wasm_check: no frame"

let pop ctx st =
  let f = top_frame st in
  if st.height = f.height then
    if f.unreachable then None else throw ctx Reason.Stack_underflow
  else
    match st.vals with
    | v :: rest ->
        st.vals <- rest;
        st.height <- st.height - 1;
        v
    | [] -> throw ctx Reason.Stack_underflow

let pop_expect ctx st expected =
  match pop ctx st with
  | None -> ()
  | Some a when Wasm_type.equal a expected -> ()
  | Some a -> throw ctx (Reason.Operand_mismatch { expected; actual = Some a })

(* Operands are listed deepest first, so they pop in reverse. *)
let pop_all ctx st ts = List.iter (pop_expect ctx st) (List.rev ts)
let push_all st ts = List.iter (fun t -> push st (Some t)) ts

let push_frame st ~label ~results =
  st.frames <-
    { label; results; height = st.height; unreachable = false } :: st.frames

let pop_frame ctx st =
  let f = top_frame st in
  pop_all ctx st f.results;
  if st.height <> f.height then
    throw ctx
      (Reason.Stack_height_mismatch { expected = f.height; actual = st.height });
  st.frames <- List.tl st.frames

let set_unreachable st =
  let f = top_frame st in
  let rec drop n l = if n <= 0 then l else drop (n - 1) (List.tl l) in
  st.vals <- drop (st.height - f.height) st.vals;
  st.height <- f.height;
  f.unreachable <- true

let block_results : Wasm.Block_type.t -> Wasm_type.t list = function
  | None -> []
  | Some t -> [ t ]

let label_frame ctx st n =
  match List.nth_opt st.frames n with
  | Some f when n >= 0 -> f
  | _ -> throw ctx (Reason.Bad_label n)

let check_memarg ctx ~natural (m : Wasm.Mem_arg.t) =
  if not ctx.has_memory then throw ctx Reason.Memory_required;
  if m.Wasm.Mem_arg.align < 0 || m.Wasm.Mem_arg.align > natural then
    throw ctx
      (Reason.Alignment_too_large { align = m.Wasm.Mem_arg.align; natural });
  if m.Wasm.Mem_arg.offset < 0 || m.Wasm.Mem_arg.offset > max_offset then
    throw ctx (Reason.Offset_out_of_range m.Wasm.Mem_arg.offset)

let local_type ctx n =
  if n >= 0 && n < Array.length ctx.locals then ctx.locals.(n)
  else throw ctx (Reason.Unknown_local n)

let global ctx n =
  if n >= 0 && n < Array.length ctx.globals then ctx.globals.(n)
  else throw ctx (Reason.Unknown_global n)

let rec instr ctx st (i : Wasm.Instr.t) =
  match i with
  | Wasm.Instr.Block (bt, body) ->
      let results = block_results bt in
      push_frame st ~label:results ~results;
      List.iter (instr ctx st) body;
      pop_frame ctx st;
      push_all st results
  | Wasm.Instr.Br n ->
      let f = label_frame ctx st n in
      pop_all ctx st f.label;
      set_unreachable st
  | Wasm.Instr.Br_if n ->
      pop_expect ctx st Wasm_type.I32;
      let f = label_frame ctx st n in
      pop_all ctx st f.label;
      push_all st f.label
  | Wasm.Instr.Call n ->
      if n < 0 || n >= Array.length ctx.func_types then
        throw ctx (Reason.Unknown_func n);
      let t = ctx.func_types.(n) in
      pop_all ctx st t.Wasm.Func_type.params;
      push_all st t.Wasm.Func_type.results
  | Wasm.Instr.Drop -> ignore (pop ctx st)
  | Wasm.Instr.F32_const _ -> push st (Some Wasm_type.F32)
  | Wasm.Instr.F64_const _ -> push st (Some Wasm_type.F64)
  | Wasm.Instr.Global_get n -> push st (Some (global ctx n).Wasm.Global.type_)
  | Wasm.Instr.Global_set n ->
      let g = global ctx n in
      if not g.Wasm.Global.mutable_ then throw ctx (Reason.Global_immutable n);
      pop_expect ctx st g.Wasm.Global.type_
  | Wasm.Instr.I32_const _ -> push st (Some Wasm_type.I32)
  | Wasm.Instr.I64_const _ -> push st (Some Wasm_type.I64)
  | Wasm.Instr.If (bt, yes, no) ->
      pop_expect ctx st Wasm_type.I32;
      let results = block_results bt in
      push_frame st ~label:results ~results;
      List.iter (instr ctx st) yes;
      pop_frame ctx st;
      push_frame st ~label:results ~results;
      List.iter (instr ctx st) no;
      pop_frame ctx st;
      push_all st results
  | Wasm.Instr.Load (l, m) ->
      check_memarg ctx ~natural:(Wasm.Load.natural_align l) m;
      pop_expect ctx st Wasm_type.I32;
      push st (Some (Wasm.Load.value_type l))
  | Wasm.Instr.Local_get n -> push st (Some (local_type ctx n))
  | Wasm.Instr.Local_set n -> pop_expect ctx st (local_type ctx n)
  | Wasm.Instr.Local_tee n ->
      let t = local_type ctx n in
      pop_expect ctx st t;
      push st (Some t)
  | Wasm.Instr.Loop (bt, body) ->
      let results = block_results bt in
      push_frame st ~label:[] ~results;
      List.iter (instr ctx st) body;
      pop_frame ctx st;
      push_all st results
  | Wasm.Instr.Memory_copy | Wasm.Instr.Memory_fill ->
      if not ctx.has_memory then throw ctx Reason.Memory_required;
      pop_all ctx st [ Wasm_type.I32; Wasm_type.I32; Wasm_type.I32 ]
  | Wasm.Instr.Numeric op ->
      let params, results = Wasm_op.signature op in
      pop_all ctx st params;
      push_all st results
  | Wasm.Instr.Return ->
      let f = List.nth st.frames (List.length st.frames - 1) in
      pop_all ctx st f.results;
      set_unreachable st
  | Wasm.Instr.Select -> (
      pop_expect ctx st Wasm_type.I32;
      let a = pop ctx st in
      let b = pop ctx st in
      match (a, b) with
      | Some x, Some y when not (Wasm_type.equal x y) ->
          throw ctx (Reason.Select_operands_differ (y, x))
      | Some x, _ | _, Some x -> push st (Some x)
      | None, None -> push st None)
  | Wasm.Instr.Store (s, m) ->
      check_memarg ctx ~natural:(Wasm.Store.natural_align s) m;
      pop_expect ctx st (Wasm.Store.value_type s);
      pop_expect ctx st Wasm_type.I32
  | Wasm.Instr.Unreachable -> set_unreachable st

let func ctx ~index (f : Wasm.Func.t) =
  let ft = f.Wasm.Func.type_ in
  let ctx =
    {
      ctx with
      func = Some index;
      locals = Array.of_list (ft.Wasm.Func_type.params @ f.Wasm.Func.locals);
    }
  in
  let results = ft.Wasm.Func_type.results in
  let st = { vals = []; height = 0; frames = [] } in
  push_frame st ~label:results ~results;
  List.iter (instr ctx st) f.Wasm.Func.body;
  pop_frame ctx st

let const_matches ctx (g : Wasm.Global.t) =
  match (g.Wasm.Global.init, g.Wasm.Global.type_) with
  | Wasm.Instr.I32_const _, Wasm_type.I32
  | Wasm.Instr.I64_const _, Wasm_type.I64
  | Wasm.Instr.F32_const _, Wasm_type.F32
  | Wasm.Instr.F64_const _, Wasm_type.F64 ->
      ()
  | ( ( Wasm.Instr.I32_const _ | Wasm.Instr.I64_const _ | Wasm.Instr.F32_const _
      | Wasm.Instr.F64_const _ ),
      expected ) ->
      let actual =
        match g.Wasm.Global.init with
        | Wasm.Instr.I32_const _ -> Wasm_type.I32
        | Wasm.Instr.I64_const _ -> Wasm_type.I64
        | Wasm.Instr.F32_const _ -> Wasm_type.F32
        | _ -> Wasm_type.F64
      in
      throw ctx (Reason.Operand_mismatch { expected; actual = Some actual })
  | _ -> throw ctx Reason.Const_expected

let max_pages_spec = 65536

let module_ (m : Wasm.Module.t) =
  Err.Escape.with_escape (fun esc ->
      let imports = m.Wasm.Module.imports in
      let n_imports = List.length imports in
      let func_types =
        Array.of_list
          (List.map (fun i -> i.Wasm.Import.type_) imports
          @ List.map (fun f -> f.Wasm.Func.type_) m.Wasm.Module.funcs)
      in
      let ctx =
        {
          func_types;
          globals = Array.of_list m.Wasm.Module.globals;
          has_memory = Option.is_some m.Wasm.Module.memory;
          locals = [||];
          esc;
          func = None;
        }
      in
      (match m.Wasm.Module.memory with
      | None -> ()
      | Some { Wasm.Memory.min_pages; max_pages } ->
          let bad =
            min_pages < 0 || min_pages > max_pages_spec
            ||
            match max_pages with
            | None -> false
            | Some x -> x < min_pages || x > max_pages_spec
          in
          if bad then throw ctx (Reason.Bad_limits { min_pages; max_pages }));
      List.iter (const_matches ctx) m.Wasm.Module.globals;
      List.iteri
        (fun k f -> func ctx ~index:(n_imports + k) f)
        m.Wasm.Module.funcs;
      let seen = Hashtbl.create 8 in
      List.iter
        (fun (e : Wasm.Export.t) ->
          if Hashtbl.mem seen e.Wasm.Export.name then
            throw ctx (Reason.Duplicate_export e.Wasm.Export.name);
          Hashtbl.add seen e.Wasm.Export.name ();
          match e.Wasm.Export.kind with
          | Wasm.Export.Func n ->
              if n < 0 || n >= Array.length func_types then
                throw ctx (Reason.Unknown_export_target n)
          | Wasm.Export.Global n ->
              if n < 0 || n >= Array.length ctx.globals then
                throw ctx (Reason.Unknown_export_target n)
          | Wasm.Export.Memory ->
              if not ctx.has_memory then throw ctx Reason.Memory_required)
        m.Wasm.Module.exports;
      List.iter
        (fun (d : Wasm.Data.t) ->
          match m.Wasm.Module.memory with
          | None -> throw ctx Reason.Memory_required
          | Some { Wasm.Memory.min_pages; _ } ->
              let offset = d.Wasm.Data.offset in
              let length = String.length d.Wasm.Data.bytes in
              if offset < 0 || offset > max_offset then
                throw ctx (Reason.Offset_out_of_range offset);
              let limit =
                Int64.mul (Int64.of_int min_pages)
                  (Int64.of_int Wasm.Memory.page_bytes)
              in
              let stop =
                Int64.add (Int64.of_int offset) (Int64.of_int length)
              in
              if Int64.compare stop limit > 0 then
                throw ctx (Reason.Data_out_of_memory { offset; length }))
        m.Wasm.Module.data)
