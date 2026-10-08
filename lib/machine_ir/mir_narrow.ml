(* In-domain index arithmetic at 32 bits. A [narrow] to 32 bits of an i64
   addition, subtraction or multiplication equals the same operation at 32
   bits on the operands truncated — two's-complement arithmetic is a ring
   homomorphism modulo 2^32 — and a truncated sign extension is the extended
   value itself, so the rewrite reads the 32-bit operands directly and drops
   the extensions it leaves unread. Only the narrowing's own defect is lost:
   the lowering emits it for operations the source proved in the index
   domain. Runs after address hoisting, which reads through the extensions
   this removes. The result is verified again. *)

(* Fault injection for the evidence suite. No consumer passes one. *)
module Mutation = struct
  type t =
    | Shifted_constant
        (** a narrowed operation's constant operand one larger *)
end

let func ?mutation (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let next_value, next_instr = Mir_select.watermarks f in
  let next_value = ref next_value and next_instr = ref next_instr in
  let vkey (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id in
  let defs = Hashtbl.create 64 in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          List.iter
            (fun v -> Hashtbl.replace defs (vkey v) i.Mir_instr.op)
            i.Mir_instr.results)
        b.Mir_block.body)
    f.Mir_func.blocks;
  let emit out (origin : Mir_origin.t) op =
    let r =
      { Mir_value.id = Mir_id.Value.of_int !next_value; ty = Mir_type.i32 }
    in
    incr next_value;
    out :=
      {
        Mir_instr.id = Mir_id.Instr.of_int !next_instr;
        results = [ r ];
        op;
        order = None;
        origin;
      }
      :: !out;
    incr next_instr;
    r
  in
  (* an i64 operand at 32 bits *)
  let narrow out origin (x : Mir_value.t) =
    let v =
      match Hashtbl.find_opt defs (vkey x) with
      | Some (Mir_op.Iext (_, Mir_width.W64, a))
        when Mir_type.equal a.Mir_value.ty Mir_type.i32 ->
          a
      | Some (Mir_op.Const c) ->
          emit out origin (Mir_op.Const (Mir_const.i32 c.Mir_const.bits))
      | _ -> emit out origin (Mir_op.Itrunc (Mir_width.W32, x))
    in
    match (mutation, Hashtbl.find_opt defs (vkey v)) with
    | Some Mutation.Shifted_constant, Some (Mir_op.Const c) ->
        emit out origin
          (Mir_op.Const (Mir_const.i32 (Int64.succ c.Mir_const.bits)))
    | _ -> v
  in
  let blocks =
    List.map
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        let body =
          List.concat_map
            (fun (i : Mir_op.t Mir_instr.t) ->
              match i.Mir_instr.op with
              | Mir_op.Narrow (Mir_width.W32, x) -> (
                  match Hashtbl.find_opt defs (vkey x) with
                  | Some
                      (Mir_op.Iarith
                         ( (( Mir_op.Iarith.Add | Mir_op.Iarith.Mul
                            | Mir_op.Iarith.Sub ) as o),
                           a,
                           b ))
                    when Mir_type.equal x.Mir_value.ty Mir_type.i64 ->
                      let out = ref [] in
                      let origin = i.Mir_instr.origin in
                      let a = narrow out origin a and b = narrow out origin b in
                      List.rev
                        ({ i with Mir_instr.op = Mir_op.Iarith (o, a, b) }
                        :: !out)
                  | _ -> [ i ])
              | _ -> [ i ])
            b.Mir_block.body
        in
        { b with Mir_block.body })
      f.Mir_func.blocks
  in
  { f with Mir_func.blocks = Mir_offsets.dead_code blocks }

let program ?mutation (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  Err.or_raise ~pp_error:Mir_diagnostic.pp
    (Mir_verify.generic
       {
         p with
         Mir_program.funcs = List.map (func ?mutation) p.Mir_program.funcs;
       })
