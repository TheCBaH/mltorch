open Ssa_bridge
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures
module Pcg = Walk_core.Pcg

(* The scalar core: constants, conversions, float and integer arithmetic (the
   failing integer division and float-to-integer cast included), comparisons,
   selection, checked index arithmetic, and loads and stores through [Coord]
   accesses on two small binary32 buffers. On top of it, structured control
   (loops carrying binary64 and int64 values, branches, ordered sums, to a
   nesting depth of two, with constant or computed bounds that may make a range
   empty) and the memory and accounting operations that fail with a row of their
   own: a scratch local, the checks of a local, a scan projection and a gather
   index, and the scan meter under limits the program is built with. No vectors,
   no integer or bfloat16 buffers: those are later families. *)

(* The operations an emitter may not have a form for: a typed refusal there is a
   capability gap, not a disagreement, and a backend that cannot express one is
   driven with a profile that leaves it out. *)
type profile = { fma : bool; i64_to_f32 : bool }

let full = { fma = true; i64_to_f32 = true }

type case = {
  seed : int64;
  size : int;
  program : Ssa_program.t;
  input : float array;
}

let default_size = 6

(* The seeds through the interpreters. [MLTORCH_FUZZ_SEEDS] sets it, and each
   emitter driver takes a fixed fraction of it (an emitted run costs a compile
   or a node start), so one variable rescales the whole suite and CI never
   raises it. *)
let seeds =
  match Option.bind (Sys.getenv_opt "MLTORCH_FUZZ_SEEDS") int_of_string_opt with
  | Some n when n > 0 -> n
  | _ -> 2000

let cells = 8
let input_id = 0
let output_id = 1

let buffers =
  let w = Int64.of_int cells in
  [
    Fx.buffer input_id ~h:1L ~w Ssa_format.F32 Ssa_buffer.Input;
    Fx.buffer output_id ~h:1L ~w Ssa_format.F32 Ssa_buffer.Output;
  ]

(* ---- the random stream ----------------------------------------------------- *)

(* Draws are [Pcg.next]'s [0, 2^32) as [int64], narrowed to [int] only after
   the remainder by a bound below 2^30, so every runtime holds them. *)
type g = { mutable pcg : Pcg.t }

let draw g bound =
  assert (bound >= 1 && bound <= 1 lsl 29);
  let n, pcg = Pcg.next g.pcg in
  g.pcg <- pcg;
  Int64.to_int (Int64.rem n (Int64.of_int bound))

let draw64 g =
  let a, pcg = Pcg.next g.pcg in
  let b, pcg = Pcg.next pcg in
  g.pcg <- pcg;
  Int64.logor (Int64.shift_left a 32) b

let pick g a = a.(draw g (Array.length a))
let chance g n = draw g n = 0

(* ---- constants ------------------------------------------------------------- *)

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Values at the edges of the rules under test: signed zeros, the binary32
   range and its overflow, the integer conversion's bounds, NaN and infinities. *)
let float_specials =
  [|
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    2.5;
    -3.75;
    0.1;
    1e300;
    -1e300;
    5e-324;
    3.4028235e38;
    3.5e38;
    16777217.;
    9.3e18;
    -9.3e18;
    9223372036854775808.;
    -9223372036854775808.;
    9223372036854774784.;
    infinity;
    neg_infinity;
    nan;
  |]

let i64_specials =
  [|
    0L;
    1L;
    -1L;
    2L;
    -2L;
    7L;
    -7L;
    Int64.min_int;
    Int64.max_int;
    16777217L;
    2147483648L;
    -2147483649L;
  |]

(* [2^k + 2^(k-24) + 1], 54 <= k <= 62: above a binary32 halfway point by less
   than a binary64 ulp (the +1 is below half an ulp of 2^k once k >= 54), so a
   conversion through binary64 sees an exact tie and rounds the other way from
   a direct one. *)
let tie_above g =
  let k = 54 + draw g 9 in
  Int64.add
    (Int64.add (Int64.shift_left 1L k) (Int64.shift_left 1L (k - 24)))
    1L

let index_specials = [| 0; 1; 2; 3; 7; -1; -8; 100; 0x7FFF_FFFF; -0x8000_0000 |]
let random_float g = float_of_int (draw g 128 - 64) /. 8.

(* ---- typed expressions ------------------------------------------------------ *)

(* What one program shares across its regions: the single scratch local it may
   allocate (with its slot count), allocated once at the top level. *)
type st = {
  mutable local : (Ssa_type.local B.value * int) option;
  var : Expr.Local_var.t;
}

(* A region's view: the builder to emit into, how many regions may still nest,
   and the binary64 and index values in scope (the carried values and the
   induction value of the enclosing loops). *)
type c = {
  g : g;
  b : B.t;
  profile : profile;
  st : st;
  nest : int;
  f64s : Ssa_type.f64 B.value list;
  idxs : Ssa_type.index B.value list;
}

let pick_list g l = List.nth l (draw g (List.length l))

(* A region's context: the new builder, one level deeper, and the values the
   region brings into scope. *)
let region c bld ?k ?(f64s = []) () =
  {
    c with
    b = bld;
    nest = c.nest - 1;
    idxs = (match k with Some k -> k :: c.idxs | None -> c.idxs);
    f64s = f64s @ c.f64s;
  }

(* The statement a region may run before it returns, set below: the loop and
   branch generators are defined before the statements they contain. *)
let effect_ref : (c -> int -> unit) ref = ref (fun _ _ -> ())
let cell_access c w = Fx.at c.b ~h:(B.index c.b 0L) ~w

let rec index c d : Ssa_type.index B.value =
  if d <= 0 || chance c.g 3 then index_leaf c
  else
    match draw c.g 7 with
    | 0 ->
        let x = index c (d - 1) in
        let y = index c (d - 1) in
        B.index_add c.b x y
    | 1 ->
        let k = pick c.g [| 2L; 3L; -1L; 0L; 65536L |] in
        B.index_scale c.b k (index c (d - 1))
    | 2 ->
        let x = index c (d - 1) in
        let y = index c (d - 1) in
        B.index_min c.b x y
    | 3 ->
        let x = index c (d - 1) in
        let y = index c (d - 1) in
        B.index_max c.b x y
    | 4 ->
        let k = pick c.g [| 1L; 2L; 3L; 7L |] in
        B.index_floor_div c.b k (index c (d - 1))
    | 5 ->
        let k = pick c.g [| 1L; 2L; 3L; 7L |] in
        B.index_ceil_div c.b k (index c (d - 1))
    | _ -> B.index_clamp_low c.b (index c (d - 1))

and index_leaf c =
  if c.idxs <> [] && chance c.g 3 then pick_list c.g c.idxs
  else if chance c.g 4 then B.index c.b (Int64.of_int (pick c.g index_specials))
  else B.index c.b (Int64.of_int (draw c.g 10))

let compare_op g = if chance g 2 then Ssa_op.Compare.Eq else Ssa_op.Compare.Lt

let rec f64 c d : Ssa_type.f64 B.value =
  if d <= 0 || chance c.g 4 then f64_leaf c
  else
    match draw c.g 12 with
    | 0 | 1 ->
        let op = pick c.g Expr.Value.[| Add; Div; Mul; Sub |] in
        let x = f64 c (d - 1) in
        let y = f64 c (d - 1) in
        B.f64_binary c.b op x y
    | 2 ->
        let x = f64 c (d - 1) in
        let y = f64 c (d - 1) in
        let z = f64 c (d - 1) in
        if c.profile.fma then B.f64_fma c.b x y z
        else
          B.f64_binary c.b Expr.Value.Add
            (B.f64_binary c.b Expr.Value.Mul x y)
            z
    | 3 ->
        let x = f64 c (d - 1) in
        let y = f64 c (d - 1) in
        B.f64_max c.b x y
    | 4 ->
        let op = pick c.g Expr.Value.[| Sqrt; Trunc |] in
        B.f64_unary c.b op (f64 c (d - 1))
    | 5 ->
        let p = pred c (d - 1) in
        let x = f64 c (d - 1) in
        let y = f64 c (d - 1) in
        B.select c.b p x y
    | 6 -> B.f32_to_f64 c.b (f32 c (d - 1))
    | 7 -> B.i64_to_f64 c.b (i64 c (d - 1))
    | 8 -> B.index_to_f64 c.b (index c (d - 1))
    | 9 -> load c (d - 1)
    | _ -> control c d

and f64_leaf c =
  if c.f64s <> [] && chance c.g 3 then pick_list c.g c.f64s
  else if chance c.g 3 then B.f64 c.b (pick c.g float_specials)
  else if chance c.g 3 then load c 0
  else B.f64 c.b (random_float c.g)

(* Structured control as a binary64 expression: a loop carrying a binary64 (and
   sometimes an int64), a branch, an ordered sum. Bounds are constants or
   computed, and may make the range empty. A region may run a store first. *)
and control c d =
  if c.nest <= 0 then load c (d - 1)
  else match draw c.g 3 with 0 -> loop c d | 1 -> branch c d | _ -> sum c d

(* What a region returns: often another region, so they nest. *)
and body c d = if chance c.g 3 then control c (d - 1) else f64 c (d - 1)

and bounds c =
  let lo = B.index c.b (Int64.of_int (draw c.g 3)) in
  let hi =
    if chance c.g 3 then
      B.index_min c.b (B.index_clamp_low c.b (index c 1)) (B.index c.b 4L)
    else B.index c.b (Int64.of_int (draw c.g 5))
  in
  (lo, hi)

and loop c d =
  let lo, hi = bounds c in
  let init = f64 c (d - 1) in
  if chance c.g 2 then
    let (B.Cons (r, B.Nil)) =
      B.for_ c.b ~lo ~hi
        ~init:(B.Cons (init, B.Nil))
        (fun bld k (B.Cons (acc, B.Nil)) ->
          let c' = region c bld ~k ~f64s:[ acc ] () in
          !effect_ref c' d;
          B.Cons (body c' d, B.Nil))
    in
    r
  else
    let init_n = i64 c (d - 1) in
    let (B.Cons (r, B.Cons (n, B.Nil))) =
      B.for_ c.b ~lo ~hi
        ~init:(B.Cons (init, B.Cons (init_n, B.Nil)))
        (fun bld k (B.Cons (acc, B.Cons (n, B.Nil))) ->
          let c' = region c bld ~k ~f64s:[ acc ] () in
          let n' = B.i64_arith bld Ssa_op.I64_op.Add n (B.index_to_i64 bld k) in
          B.Cons (body c' d, B.Cons (n', B.Nil)))
    in
    B.f64_binary c.b Expr.Value.Add r (B.i64_to_f64 c.b n)

and branch c d =
  let p = pred c (d - 1) in
  let (B.Cons (r, B.Nil)) =
    B.if_ c.b p
      ~then_:(fun bld ->
        let c' = region c bld () in
        !effect_ref c' d;
        B.Cons (body c' d, B.Nil))
      ~else_:(fun bld ->
        let c' = region c bld () in
        B.Cons (body c' d, B.Nil))
  in
  r

and sum c d =
  let lo, hi = bounds c in
  B.ordered_sum c.b ~lo ~hi
    ~seed:(f64 c (d - 1))
    (fun bld k ->
      let c' = region c bld ~k () in
      if chance c.g 3 then B.mark bld Ssa_mark.Reduction;
      body c' d)

(* A load from the input: in range most of the time, a raw index (which may
   leave the buffer, and then fails with its row) otherwise. *)
and load c d =
  let w =
    if chance c.g 12 then index c d
    else
      let x = index c d in
      B.index_min c.b (B.index_clamp_low c.b x) (B.index c.b 7L)
  in
  B.load_f64 c.b (Fx.buf input_id) ~decode:Ssa_op.Decode.F32_to_f64
    (cell_access c w)

and f32 c d : Ssa_type.f32 B.value =
  match draw c.g 3 with
  | 0 -> B.f64_to_f32 c.b (f64 c (d - 1))
  | 1 -> i64_to_f32 c (i64 c (d - 1))
  | _ -> B.f32 c.b (to_f32 (random_float c.g))

and i64_to_f32 c x =
  if c.profile.i64_to_f32 then B.i64_to_f32 c.b x
  else B.f64_to_f32 c.b (B.i64_to_f64 c.b x)

and i64 c d : Ssa_type.i64 B.value =
  if d <= 0 || chance c.g 3 then i64_leaf c
  else
    match draw c.g 6 with
    | 0 | 1 ->
        let op = pick c.g Ssa_op.I64_op.[| Add; Mul; Sub |] in
        let x = i64 c (d - 1) in
        let y = i64 c (d - 1) in
        B.i64_arith c.b op x y
    | 2 ->
        let x = i64 c (d - 1) in
        let y = i64 c (d - 1) in
        B.i64_div c.b x y
    | 3 -> B.float_to_i64 c.b (f64 c (d - 1))
    | 4 -> B.index_to_i64 c.b (index c (d - 1))
    | _ ->
        let p = pred c (d - 1) in
        let x = i64 c (d - 1) in
        let y = i64 c (d - 1) in
        B.select c.b p x y

and i64_leaf c =
  match draw c.g 4 with
  | 0 -> B.i64 c.b (pick c.g i64_specials)
  | 1 -> B.i64 c.b (tie_above c.g)
  | 2 -> B.i64 c.b (draw64 c.g)
  | _ -> B.i64 c.b (Int64.of_int (draw c.g 16 - 8))

and pred c d : Ssa_type.pred B.value =
  if d <= 0 || chance c.g 4 then B.pred c.b (chance c.g 2)
  else
    match draw c.g 5 with
    | 0 ->
        let op = compare_op c.g in
        let x = f64 c (d - 1) in
        let y = f64 c (d - 1) in
        B.float_compare c.b op x y
    | 1 ->
        let op = compare_op c.g in
        let x = i64 c (d - 1) in
        let y = i64 c (d - 1) in
        B.i64_compare c.b op x y
    | 2 ->
        let op = compare_op c.g in
        let x = index c (d - 1) in
        let y = index c (d - 1) in
        B.index_compare c.b op x y
    | 3 -> B.pred_not c.b (pred c (d - 1))
    | _ ->
        let x = pred c (d - 1) in
        let y = pred c (d - 1) in
        B.pred_or c.b x y

(* ---- statements ------------------------------------------------------------- *)

let store c w x =
  B.store_f64 c.b (Fx.buf output_id) ~encode:Ssa_op.Encode.F32_round
    (cell_access c w) x

(* Targeted idioms: each pins one rule to a value that shows it, so a seed range
   reaches every failure row and every rounding and comparison the emitters can
   get wrong without leaving it to chance. *)
let probes =
  [|
    (fun c ->
      B.i64_to_f64 c.b
        (B.i64_div c.b (B.i64 c.b Int64.min_int) (B.i64 c.b (-1L))));
    (fun c -> B.i64_to_f64 c.b (B.i64_div c.b (B.i64 c.b 7L) (B.i64 c.b 0L)));
    (fun c -> B.i64_to_f64 c.b (B.i64_div c.b (B.i64 c.b (-7L)) (B.i64 c.b 2L)));
    (fun c ->
      B.i64_to_f64 c.b
        (B.float_to_i64 c.b (B.f64 c.b (pick c.g float_specials))));
    (fun c -> B.f32_to_f64 c.b (i64_to_f32 c (B.i64 c.b (tie_above c.g))));
    (fun c ->
      let p =
        B.i64_compare c.b Ssa_op.Compare.Lt (B.i64 c.b (-1L)) (B.i64 c.b 1L)
      in
      B.select c.b p (B.f64 c.b 1.) (B.f64 c.b 0.));
    (fun c ->
      let top = B.index c.b 0x7FFF_FFFFL in
      B.index_to_f64 c.b (B.index_add c.b top (B.index c.b 1L)));
    (fun c ->
      let w = B.index c.b (if chance c.g 2 then 8L else -1L) in
      B.load_f64 c.b (Fx.buf input_id) ~decode:Ssa_op.Decode.F32_to_f64
        (cell_access c w));
    (fun c ->
      let x = B.f64 c.b (pick c.g float_specials) in
      let y = B.f64 c.b (pick c.g float_specials) in
      B.f64_binary c.b Expr.Value.Sub x y);
  |]

(* A store outside its buffer is a defect of the program, not a row: only a
   checked access fails, so a computed cell is checked first and a constant one
   is in range. *)
let statement c d =
  let w =
    if chance c.g 10 then (
      let w = index c 1 in
      B.check_access c.b (Fx.buf output_id) (cell_access c w);
      w)
    else B.index c.b (Int64.of_int (draw c.g cells))
  in
  let x = if chance c.g 8 then (pick c.g probes) c else f64 c d in
  store c w x

let () = effect_ref := statement

(* Effect-only regions: stores inside a branch or a loop body that carries
   nothing. *)
let if_statement c d =
  let p = pred c d in
  ignore
    (B.if_ c.b p
       ~then_:(fun bld ->
         statement (region c bld ()) d;
         B.Nil)
       ~else_:(fun bld ->
         if chance c.g 2 then statement (region c bld ()) d;
         B.Nil))

let loop_statement c d =
  let lo, hi = bounds c in
  ignore
    (B.for_ c.b ~lo ~hi ~init:B.Nil (fun bld k B.Nil ->
         statement (region c bld ~k ()) d;
         B.Nil))

(* One scratch local per program, allocated at the top level with every slot
   written, so a read never meets an unset cell. A write stays in range; a read
   may leave it, and then fails with the unbound-local row. *)
let local_statement c d =
  let h, slots =
    match c.st.local with
    | Some l -> l
    | None ->
        let slots = 1 + draw c.g 3 in
        let h = B.local_alloc ~var:c.st.var c.b ~slots:(Int64.of_int slots) in
        for i = 0 to slots - 1 do
          B.local_write c.b h (B.index c.b (Int64.of_int i)) (f64 c 1)
        done;
        c.st.local <- Some (h, slots);
        (h, slots)
  in
  let clamped () =
    B.index_min c.b
      (B.index_clamp_low c.b (index c 1))
      (B.index c.b (Int64.of_int (slots - 1)))
  in
  if chance c.g 2 then B.local_write c.b h (clamped ()) (f64 c d)
  else
    let at = if chance c.g 6 then index c 1 else clamped () in
    let w = B.index c.b (Int64.of_int (draw c.g cells)) in
    store c w (B.local_read c.b h at)

(* The checks that fail with a row of their own, on positions that are mostly
   outside their extents. *)
let check_statement c _d =
  match draw c.g 3 with
  | 0 -> B.check_local c.b ~var:c.st.var ~extent:2L (index c 1)
  | 1 ->
      let var = if chance c.g 2 then Some c.st.var else None in
      B.check_scan c.b ~var ~row:(index c 1) ~lane:(index c 1) ~row_extent:3L
        ~lane_extent:2L
  | _ -> B.check_gather c.b (i64 c 1) ~extent:4L

(* The scan meter: charges, reservations that are released, a reset. The limits
   the program was built with decide which of them fail. *)
let meter_statement c =
  let width () = Int64.of_int (1 + draw c.g 3) in
  match draw c.g 4 with
  | 0 ->
      for _ = 0 to draw c.g 4 do
        B.meter_charge c.b
      done
  | 1 ->
      let w = width () in
      B.meter_reserve c.b ~width:w;
      if chance c.g 2 then B.meter_charge c.b;
      B.meter_release c.b ~width:w
  | 2 -> B.meter_reserve c.b ~width:(width ())
  | _ -> B.meter_reset c.b

let top_statement c d =
  match draw c.g 20 with
  | 12 -> if_statement c d
  | 13 -> loop_statement c d
  | 14 | 15 -> local_statement c d
  | 16 -> check_statement c d
  | 17 -> meter_statement c
  | _ -> statement c d

let local_var = Expr.Builder.run Expr.Builder.fresh_local

let limits g =
  if chance g 2 then None
  else
    let max_state = 2 + draw g 7 and max_updates = Int64.of_int (draw g 6) in
    match Err.payload (Expr.Scan_limits.create ~max_state ~max_updates) with
    | Ok l -> Some l
    | Error _ -> None

let generate ?(profile = full) ~seed ~size () =
  let g = { pcg = Pcg.seed ~seed ~seq:0x5eedL } in
  let input =
    Array.init cells (fun _ ->
        if chance g 12 then pick g [| 0.; -0.; infinity; neg_infinity; nan |]
        else to_f32 (random_float g))
  in
  let depth = min 4 (1 + (size / 2)) in
  let scan_limits = limits g in
  let build bld =
    let c =
      {
        g;
        b = bld;
        profile;
        st = { local = None; var = local_var };
        nest = 2;
        f64s = [];
        idxs = [];
      }
    in
    for _ = 1 to 1 + size do
      top_statement c depth
    done
  in
  match Err.payload (B.program ?scan_limits ~buffers build) with
  | Ok program -> { seed; size; program; input }
  | Error e ->
      Fmt.failwith
        "the generator built a program that does not verify (seed=%Ld \
         size=%d): %a"
        seed size Ssa_verify.pp_error e

(* ---- running ---------------------------------------------------------------- *)

type outcome = { row : Kernel_eval.error option; cells : float list }

let equal_outcome a b =
  a.row = b.row
  && List.length a.cells = List.length b.cells
  && List.for_all2 Core.Float_bits.equal_portable a.cells b.cells

(* A NaN's sign is the host's: x86 makes -nan where arm64 makes nan. *)
let pp_cell ppf x =
  if Float.is_nan x then Fmt.string ppf "nan" else Fmt.float ppf x

let pp_outcome ppf o =
  match o.row with
  | Some e -> Fmt.pf ppf "row: %a" Kernel_eval.pp_error e
  | None -> Fmt.pf ppf "ok: %a" Fmt.(list ~sep:(any " ") pp_cell) o.cells

type run = Ran of outcome | Refused of string
type backend = { name : string; run : case -> run }

let memory case =
  let out = Array.make cells 0. in
  let m =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let cs =
          if (b.Ssa_buffer.id :> int) = input_id then Array.copy case.input
          else out
        in
        Ssa_id.Buffer.Map.add b.Ssa_buffer.id (Ssa_memory.Floats cs) m)
      Ssa_id.Buffer.Map.empty case.program.Ssa_program.buffers
  in
  (m, out)

let finish row out =
  match row with
  | Some _ -> { row; cells = [] }
  | None -> { row = None; cells = Array.to_list out }

let interp =
  let run case =
    let memory, out = memory case in
    let row =
      match Err.payload (Ssa_interp.run case.program ~memory) with
      | Ok () -> None
      | Error (#Ssa_interp.failure as f) -> Some (f :> Kernel_eval.error)
      | Error (`Invalid_program _) -> failwith "the program does not verify"
    in
    Ran (finish row out)
  in
  { name = "interp"; run }

let cfg =
  let run case =
    let graph =
      match Err.payload (Ssa_cfg_lower.program case.program) with
      | Ok g -> g
      | Error e -> Fmt.failwith "%a" Ssa_cfg_lower.pp_error e
    in
    let memory, out = memory case in
    let row =
      match Err.payload (Ssa_cfg_interp.run graph ~memory) with
      | Ok () -> None
      | Error (#Ssa_interp.failure as f) -> Some (f :> Kernel_eval.error)
      | Error e -> Fmt.failwith "%a" Ssa_cfg_interp.pp_error e
    in
    Ran (finish row out)
  in
  { name = "cfg"; run }

(* What an emitted kernel runs over: the Loop program that supplies its
   argument layout, and the binding of the input buffer. The three emitters
   (C, JavaScript, Wasm) share both. *)
let emitted case arguments =
  let loop =
    Lf.program ~buffers:(List.map Loop_of_ssa.loop_buffer arguments) []
  in
  let shape = Lf.shape_w cells in
  let bind id =
    if Tensor_id.equal id (Lf.tid input_id) then
      Some
        (Lf.f32_tensor shape (fun co ->
             case.input.((Vec6.offset shape co :> int))))
    else None
  in
  (loop, bind)

let of_row (e : Loop_ir.Loop_interp.error) =
  Ran { row = Some (e :> Kernel_eval.error); cells = [] }

let of_outputs outputs =
  let cs =
    match Tensor_id.Map.find_opt (Lf.tid output_id) outputs with
    | Some t -> Lf.cells t cells
    | None -> List.init cells (fun _ -> 0.)
  in
  Ran { row = None; cells = cs }

(* ---- the differential ------------------------------------------------------- *)

type disagreement = {
  case : case;
  backend : string;
  expected : outcome;
  got : outcome;
}

let ran b case =
  match b.run case with
  | Ran o -> o
  | Refused why -> Fmt.failwith "%s refused: %s" b.name why

let check case backends =
  let expected = ran interp case in
  let rec go = function
    | [] -> Ok expected
    | b :: rest -> (
        match b.run case with
        | Refused _ -> go rest
        | Ran got ->
            if equal_outcome expected got then go rest
            else Error { case; backend = b.name; expected; got })
  in
  go backends

let minimize ?profile ~seed ~size backends =
  let rec go s =
    if s > size then None
    else
      match check (generate ?profile ~seed ~size:s ()) backends with
      | Error d -> Some d
      | Ok _ -> go (s + 1)
  in
  go 1

let pp_disagreement ppf d =
  Fmt.pf ppf
    "@[<v>%s disagrees with the interpreter at seed=%Ld size=%d@,\
     %s@,\
     interp: %a@,\
     %s: %a@]"
    d.backend d.case.seed d.case.size
    (Ssa_pp.to_string d.case.program)
    pp_outcome d.expected d.backend pp_outcome d.got

let replay ?profile ~seed ~size backends =
  let case = generate ?profile ~seed ~size () in
  Fmt.pr "seed=%Ld size=%d@.%s@." seed size (Ssa_pp.to_string case.program);
  List.iter
    (fun b ->
      match b.run case with
      | Ran o -> Fmt.pr "%s: %a@." b.name pp_outcome o
      | Refused why -> Fmt.pr "%s: refused (%s)@." b.name why)
    (interp :: backends)

let row_name : Kernel_eval.error -> string = function
  | `Coord_out_of_range _ -> "coord_out_of_range"
  | `Gather_index_out_of_range _ -> "gather_index_out_of_range"
  | `I64_division_by_zero -> "i64_division_by_zero"
  | `I64_division_overflow -> "i64_division_overflow"
  | `I64_from_float_infinite -> "i64_from_float_infinite"
  | `I64_from_float_nan -> "i64_from_float_nan"
  | `I64_from_float_out_of_range _ -> "i64_from_float_out_of_range"
  | `Index_overflow _ -> "index_overflow"
  | `Scan_meter _ -> "scan_meter"
  | `Scan_projection _ -> "scan_projection"
  | `Unbound_local _ -> "unbound_local"
  | _ -> "other"
