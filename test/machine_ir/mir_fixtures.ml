(* Hand-built generic programs shared by the verifier, printer and interpreter
   suites. Each is one function with explicit pointer and scalar parameters. *)
open Machine_ir
module B = Mir_builder

let fn = Mir_id.Func.of_int 0
let i32 = Mir_type.i32
let i64 = Mir_type.i64
let const bld blk c = B.emit bld blk (Mir_op.Const c)
let k64 bld blk x = const bld blk (Mir_const.i64 x)
let k32 bld blk x = const bld blk (Mir_const.i32 x)
let kf64 bld blk x = const bld blk (Mir_const.f64 x)

let program ?regions ?views ?helpers ?planning f =
  B.program ?regions ?views ?helpers ?planning [ f ] ~main:fn

(* (a, b) <- (b, a) [n] times, from (1, 2): the edge must rebind both at once. *)
let swap ?(sequential = false) () =
  let bld = B.create () in
  let entry = B.new_block bld [ i32 ] in
  let header = B.new_block bld [ i32; i64; i64 ] in
  let body = B.new_block bld [] in
  let exit = B.new_block bld [ i64; i64 ] in
  let n = List.hd (B.param entry) in
  let zero = k32 bld entry 0L in
  let one = k64 bld entry 1L and two = k64 bld entry 2L in
  B.jump entry header [ zero; one; two ];
  let i, a, b =
    match B.param header with [ i; a; b ] -> (i, a, b) | _ -> assert false
  in
  let go = B.emit bld header (Mir_op.Icmp (Mir_op.Icmp.Slt, i, n)) in
  B.branch header go (body, []) (exit, [ a; b ]);
  let step = k32 bld body 1L in
  let i' = B.emit bld body (Mir_op.Iarith (Mir_op.Iarith.Add, i, step)) in
  (* the sequential mutation reads the already-rebound value *)
  B.jump body header (if sequential then [ i'; b; b ] else [ i'; b; a ]);
  B.return exit (B.param exit);
  program (B.func bld ~id:fn ~name:"swap" ~entry ~results:[ i64; i64 ])

(* The sum of [len] binary32 cells at [base], each widened to binary64 and
   added in order; [i < 0 || i >= extent] fails as a coordinate failure on the
   W axis, checked before the load. Iterates [i] from 0 to [len]. *)
let sum_f32 ?(extent = 0L) ?(source = 7) () =
  let bld = B.create () in
  let entry = B.new_block bld [ Mir_type.Ptr; i32 ] in
  let header = B.new_block bld [ i32; Mir_type.F64 ] in
  let check = B.new_block bld [] in
  let access = B.new_block bld [] in
  let invalid = B.new_block bld [] in
  let exit = B.new_block bld [ Mir_type.F64 ] in
  let base, len =
    match B.param entry with [ b; l ] -> (b, l) | _ -> assert false
  in
  let zero = k32 bld entry 0L in
  let acc0 = kf64 bld entry 0. in
  B.jump entry header [ zero; acc0 ];
  let i, acc =
    match B.param header with [ i; a ] -> (i, a) | _ -> assert false
  in
  let go = B.emit bld header (Mir_op.Icmp (Mir_op.Icmp.Slt, i, len)) in
  B.branch header go (check, []) (exit, [ acc ]);
  let ext = k32 bld check (if extent = 0L then 0x7FFF_FFFFL else extent) in
  let z = k32 bld check 0L in
  let lo = B.emit bld check (Mir_op.Icmp (Mir_op.Icmp.Sle, z, i)) in
  let hi = B.emit bld check (Mir_op.Icmp (Mir_op.Icmp.Slt, i, ext)) in
  let ok = B.emit bld check (Mir_op.Pbinary (Mir_op.Pbinary.And, lo, hi)) in
  B.branch check ok (access, []) (invalid, []);
  let wide =
    B.emit bld access (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, i))
  in
  let four = k64 bld access 4L in
  let bytes =
    B.emit bld access (Mir_op.Iarith (Mir_op.Iarith.Mul, wide, four))
  in
  let addr = B.emit bld access (Mir_op.Ptr_add (base, bytes)) in
  let bits =
    B.emit bld access
      (Mir_op.Load { Mir_op.Access.width = Mir_width.W32; addr; align = 4L })
  in
  let cell = B.emit bld access (Mir_op.Bitcast (Mir_type.F32, bits)) in
  let x =
    B.emit bld access (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, cell))
  in
  let acc' = B.emit bld access (Mir_op.Fbinary (Mir_op.Fbinary.Add, acc, x)) in
  let step = k32 bld access 1L in
  let i' = B.emit bld access (Mir_op.Iarith (Mir_op.Iarith.Add, i, step)) in
  B.jump access header [ i'; acc' ];
  let wi =
    B.emit bld invalid (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, i))
  in
  let z64 = k64 bld invalid 0L in
  B.fail invalid
    (Mir_failure.Coord_out_of_range
       {
         Mir_failure.Coord.source = Expr.Source.create source;
         axis = Expr.Axis.W;
       })
    [ z64; z64; z64; z64; wi; z64 ];
  B.return exit (B.param exit);
  program (B.func bld ~id:fn ~name:"sum_f32" ~entry ~results:[ Mir_type.F64 ])

(* Checked i64 division: a zero divisor fails first, then [min / -1]. *)
let div () =
  let bld = B.create () in
  let entry = B.new_block bld [ i64; i64 ] in
  let nonzero = B.new_block bld [] in
  let zero = B.new_block bld [] in
  let ok = B.new_block bld [] in
  let overflow = B.new_block bld [] in
  let x, y =
    match B.param entry with [ x; y ] -> (x, y) | _ -> assert false
  in
  let k0 = k64 bld entry 0L in
  let is_zero = B.emit bld entry (Mir_op.Icmp (Mir_op.Icmp.Eq, y, k0)) in
  B.branch entry is_zero (zero, []) (nonzero, []);
  B.fail zero Mir_failure.I64_division_by_zero [];
  let kmin = k64 bld nonzero Int64.min_int and km1 = k64 bld nonzero (-1L) in
  let a = B.emit bld nonzero (Mir_op.Icmp (Mir_op.Icmp.Eq, x, kmin)) in
  let b = B.emit bld nonzero (Mir_op.Icmp (Mir_op.Icmp.Eq, y, km1)) in
  let both = B.emit bld nonzero (Mir_op.Pbinary (Mir_op.Pbinary.And, a, b)) in
  B.branch nonzero both (overflow, []) (ok, []);
  B.fail overflow Mir_failure.I64_division_overflow [];
  let q = B.emit bld ok (Mir_op.Idiv (Mir_op.Idiv.Div_s, x, y)) in
  B.return ok [ q ];
  program (B.func bld ~id:fn ~name:"div" ~entry ~results:[ i64 ])

(* Stores [x] as binary32 at [base + 4 * i] and an event per store. *)
let store_f32 () =
  let bld = B.create () in
  let entry = B.new_block bld [ Mir_type.Ptr; Mir_type.F64 ] in
  let base, x =
    match B.param entry with [ b; x ] -> (b, x) | _ -> assert false
  in
  let r = B.emit bld entry (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, x)) in
  let bits = B.emit bld entry (Mir_op.Bitcast (Mir_type.i32, r)) in
  B.emit_unit bld entry
    (Mir_op.Store
       ({ Mir_op.Access.width = Mir_width.W32; addr = base; align = 4L }, bits));
  B.emit_unit bld entry (Mir_op.Event (Mir_event.Emitter, 1L));
  B.return entry [];
  program (B.func bld ~id:fn ~name:"store_f32" ~entry ~results:[])
