(* The row normalization's loops, narrowed to binary32 and interpreted, give the
   bits [Norm_block.eval] gives; and the plan that contracts dots fuses exactly
   the sites the descriptor's [F32_fused] fuses. *)

open Ssa_ir
module F = Ssa_fixtures
module N = Norm_block
module P = Attention_block.Precision

let i64 = Int64.of_int

(* x is read with a row stride wider than the row, so the strides are seen *)
let pad = 3

let desc ~kind ~precision ~weight ~bias ~rows ~width =
  Result.get_ok
    (Err.payload
       (N.make ~kind ~rows:(i64 rows) ~width:(i64 width)
          ~eps:(F32_math.round32 1e-5) ~weight ~bias
          ~x:{ N.Strides.row = i64 (width + pad); col = 1L }
          ~y:{ N.Strides.row = i64 width; col = 1L }
          ~precision))

let lcg = ref 99

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let f32 id ~h ~w role = F.buffer id ~h:(i64 h) ~w:(i64 w) Ssa_format.F32 role

(* buffers 0 x, 1 weight, 2 bias, 3 y *)
let program (d : N.t) =
  let rows = Int64.to_int d.N.rows and width = Int64.to_int d.N.width in
  let buffers =
    [ f32 0 ~h:rows ~w:(width + pad) Ssa_buffer.Input ]
    @ (if d.N.weight then [ f32 1 ~h:1 ~w:width Ssa_buffer.Input ] else [])
    @ (if d.N.bias then [ f32 2 ~h:1 ~w:width Ssa_buffer.Input ] else [])
    @ [ f32 3 ~h:rows ~w:width Ssa_buffer.Output ]
  in
  Ssa_norm.program d ~x:(F.buf 0)
    ~weight:(if d.N.weight then Some (F.buf 1) else None)
    ~bias:(if d.N.bias then Some (F.buf 2) else None)
    ~y:(F.buf 3) ~buffers

let inputs (d : N.t) =
  let rows = Int64.to_int d.N.rows and width = Int64.to_int d.N.width in
  ( Array.init (rows * (width + pad)) (fun _ -> value ()),
    Array.init width (fun _ -> value ()),
    Array.init width (fun _ -> value ()) )

let reference (d : N.t) (x, w, b) =
  let out = Array.make (Int64.to_int (Int64.mul d.N.rows d.N.width)) 0. in
  let at a o = a.(Int64.to_int o) in
  N.eval d ~x:(at x) ~weight:(at w) ~bias:(at b) ~write:(fun o v ->
      out.(Int64.to_int o) <- v);
  out

let memory (d : N.t) (x, w, b) =
  F.memory
    ([ (0, F.floats (Array.copy x)) ]
    @ (if d.N.weight then [ (1, F.floats w) ] else [])
    @ (if d.N.bias then [ (2, F.floats b) ] else [])
    @ [
        ( 3,
          F.floats (Array.make (Int64.to_int (Int64.mul d.N.rows d.N.width)) 0.)
        );
      ])

let output memory =
  match Ssa_id.Buffer.Map.find (F.buf 3) memory with
  | Ssa_memory.Floats a -> a
  | _ -> assert false

let bitwise a b =
  Array.length a = Array.length b
  && Array.for_all2 Core.Float_bits.equal_portable a b

(* The structured interpreter on the exact binary32 form. *)
let interpret p d cells =
  let m = memory d cells in
  F.run (Ssa_precision.to_f32 ~exact:true p) ~memory:m;
  output m

(* What the Machine IR route makes of a plan that contracts dots: its loops'
   accumulations fused, its ordered sums lowered with each product term fused,
   run by the CFG interpreter. *)
let contracted p (d : N.t) cells =
  let plan = Ssa_plan.resolve ~numerics:Ssa_numerics.Fp32_contracted p in
  let cfg =
    match
      Ssa_cfg_lower.program ~accumulate:Ssa_cfg_lower.Accumulate.Fuse_products
        (Ssa_fuse_dots.program plan.Ssa_plan.program)
    with
    | Ok c -> c
    | Error e -> Fmt.failwith "%a" Ssa_cfg_lower.pp_error e
  in
  let m = memory d cells in
  Err.or_raise ~pp_error:Ssa_cfg_interp.pp_error
    (Ssa_cfg_interp.run cfg ~memory:m);
  output m

let shapes = [ (1, 1); (3, 4); (5, 17) ]
let affines = [ (false, false); (true, false); (true, true) ]

let%expect_test "the norm's loops equal the reference" =
  List.iter
    (fun kind ->
      List.iter
        (fun precision ->
          let fails = ref [] and count = ref 0 in
          List.iter
            (fun (rows, width) ->
              List.iter
                (fun (weight, bias) ->
                  if not (bias && kind = N.Kind.Rms) then (
                    let d = desc ~kind ~precision ~weight ~bias ~rows ~width in
                    let cells = inputs d in
                    incr count;
                    if
                      not
                        (bitwise
                           (interpret (program d) d cells)
                           (reference d cells))
                    then fails := N.name d :: !fails))
                affines)
            shapes;
          Fmt.pr "%s %s: %d cases, %a@." (N.Kind.name kind) (P.name precision)
            !count
            Fmt.(
              list ~sep:comma string |> fun pp ppf -> function
              | [] -> string ppf "all equal the reference"
              | l -> pf ppf "DIFFER: %a" pp l)
            (List.rev !fails))
        [ P.F32_ordered; P.F32_fused ])
    [ N.Kind.Layer; N.Kind.Rms ];
  [%expect
    {|
    layer f32_ordered: 9 cases, all equal the reference
    layer f32_fused: 9 cases, all equal the reference
    rms f32_ordered: 6 cases, all equal the reference
    rms f32_fused: 6 cases, all equal the reference |}]

let%expect_test "the plan that contracts dots fuses the descriptor's sites" =
  List.iter
    (fun kind ->
      List.iter
        (fun (weight, bias) ->
          if not (bias && kind = N.Kind.Rms) then
            let d precision =
              desc ~kind ~precision ~weight ~bias ~rows:32 ~width:17
            in
            let cells = inputs (d P.F32_ordered) in
            let fused = reference (d P.F32_fused) cells
            and ordered = reference (d P.F32_ordered) cells in
            Fmt.pr
              "%s w%b b%b: the two references differ %b; ordered loops \
               contracted = fused %b; fused loops contracted = fused %b; \
               ordered loops uncontracted = ordered %b@."
              (N.Kind.name kind) weight bias
              (not (bitwise fused ordered))
              (bitwise
                 (contracted
                    (program (d P.F32_ordered))
                    (d P.F32_ordered) cells)
                 fused)
              (bitwise
                 (contracted (program (d P.F32_fused)) (d P.F32_fused) cells)
                 fused)
              (bitwise
                 (interpret (program (d P.F32_ordered)) (d P.F32_ordered) cells)
                 ordered))
        affines)
    [ N.Kind.Layer; N.Kind.Rms ];
  [%expect
    {|
    layer wfalse bfalse: the two references differ true; ordered loops contracted = fused true; fused loops contracted = fused true; ordered loops uncontracted = ordered true
    layer wtrue bfalse: the two references differ true; ordered loops contracted = fused true; fused loops contracted = fused true; ordered loops uncontracted = ordered true
    layer wtrue btrue: the two references differ true; ordered loops contracted = fused true; fused loops contracted = fused true; ordered loops uncontracted = ordered true
    rms wfalse bfalse: the two references differ true; ordered loops contracted = fused true; fused loops contracted = fused true; ordered loops uncontracted = ordered true
    rms wtrue bfalse: the two references differ true; ordered loops contracted = fused true; fused loops contracted = fused true; ordered loops uncontracted = ordered true |}]

let%expect_test "a bfloat16 norm has no loops" =
  (match
     program
       (desc ~kind:N.Kind.Layer ~precision:P.Bf16_paired ~weight:false
          ~bias:false ~rows:1 ~width:2)
   with
  | _ -> Fmt.pr "built@."
  | exception Invalid_argument m -> Fmt.pr "%s@." m);
  [%expect {| Ssa_norm: a bfloat16 norm has no store |}]
