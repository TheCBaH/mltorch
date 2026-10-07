open Machine_ir
open Machine_interp
open Mir_fixtures
open Mir_harness

(* C1: generic programs executed by the independent interpreter, and the
   execution mutations each produce a mismatch or a defect. *)

let i32v x = Val (Mir_datum.i32 x)
let i64v x = Val (Mir_datum.i64 x)
let print_run (r, _, _) = print_endline (show_outcome r)

let%expect_test "swaps and zero-trip recurrences" =
  List.iter (fun n -> print_run (run (swap ()) [ i32v n ])) [ 0L; 1L; 5L; 6L ];
  (* sequential rebinding reads the already-overwritten value *)
  print_run (run (swap ~sequential:true ()) [ i32v 5L ]);
  [%expect
    {|
    success [0x1, 0x2]
    success [0x2, 0x1]
    success [0x2, 0x1]
    success [0x1, 0x2]
    success [0x2, 0x2] |}]

let sum_program ?extent n =
  with_objects (sum_f32 ?extent ())
    ~regions:[ region 0 (Int64.of_int (4 * n)) ]
    ~views:[ view 0 ~region:0 (Int64.of_int (4 * n)) ]

let%expect_test "binary32 cells summed in binary64, lazily guarded" =
  let xs = [ 0.1; 1e-8; -3.5; 16777217.; Float.ldexp 1. (-140) ] in
  let n = List.length xs in
  let p = sum_program n in
  let r, _, _ =
    run p ~bound:[ (0, f32_bytes xs) ] [ Ptr 0; i32v (Int64.of_int n) ]
  in
  (* the reference: each cell rounded to binary32, widened, added in order *)
  let expected =
    List.fold_left
      (fun acc x -> acc +. Int32.float_of_bits (Int32.bits_of_float x))
      0. xs
  in
  Fmt.pr "%s exact=%b@." (show_outcome r) (f64_result r = Some expected);
  (* the empty sum *)
  print_run (run p ~bound:[ (0, f32_bytes xs) ] [ Ptr 0; i32v 0L ]);
  (* the guard fails at i = 3 before the load: the region holds only three
     cells, so an eager load would be a bad access *)
  let p3 = sum_program ~extent:3L 3 in
  print_run (run p3 ~bound:[ (0, f32_bytes [ 1.; 2.; 3. ]) ] [ Ptr 0; i32v 5L ]);
  [%expect
    {|
    success [0x416fffff93333339] exact=true
    success [0x0]
    failure coord_out_of_range(t7, W)(0:i64, 0:i64, 0:i64, 0:i64, 3:i64, 0:i64) |}]

module B = Mir_builder

let load32 bld blk addr =
  B.emit bld blk
    (Mir_op.Load { Mir_op.Access.width = Mir_width.W32; addr; align = 4L })

let one_block ?(regions = []) ?(views = []) params results body =
  let bld = B.create () in
  let e = B.new_block bld params in
  let rs = body bld e (B.param e) in
  B.return e rs;
  with_objects
    (program (B.func bld ~id:fn ~name:"t" ~entry:e ~results))
    ~regions ~views

(* The cell at [base + scale * ext(i)], loaded as 32 bits. *)
let load_cell ?(ext = Mir_op.Iext.Sext) ?(width = Mir_width.W32) () =
  one_block [ Mir_type.Ptr; i32 ] [ Mir_type.Int width ] (fun bld e -> function
    | [ base; i ] ->
        let wide = B.emit bld e (Mir_op.Iext (ext, Mir_width.W64, i)) in
        let off =
          B.emit bld e (Mir_op.Iarith (Mir_op.Iarith.Mul, wide, k64 bld e 4L))
        in
        let addr = B.emit bld e (Mir_op.Ptr_add (base, off)) in
        [ B.emit bld e (Mir_op.Load { Mir_op.Access.width; addr; align = 4L }) ]
    | _ -> assert false)
  |> with_objects
       ~regions:[ region 0 12L ]
       ~views:[ view 0 ~region:0 12L; view 1 ~region:0 ~offset:4L 8L ]

let%expect_test "addressing: extension, access width, wide offsets" =
  let cells = f32_bytes [ 1.; 2.; 3. ] in
  (* view 1 starts at cell 1, so index -1 names cell 0 of the region; but the
     pointer's window is the view, so even the correct sign extension may not
     reach outside it *)
  print_run (run (load_cell ()) ~bound:[ (0, cells) ] [ Ptr 1; i32v 1L ]);
  print_run (run (load_cell ()) ~bound:[ (0, cells) ] [ Ptr 1; i32v (-1L) ]);
  (* mutation: zero extension turns -1 into a 16 GiB offset *)
  print_run
    (run
       (load_cell ~ext:Mir_op.Iext.Zext ())
       ~bound:[ (0, cells) ]
       [ Ptr 0; i32v (-1L) ]);
  (* mutation: a 64-bit access at the last cell crosses the end *)
  print_run
    (run
       (load_cell ~width:Mir_width.W64 ())
       ~bound:[ (0, cells) ]
       [ Ptr 0; i32v 2L ]);
  (* a region wider than 32 bits, stored sparsely: a cell past 2^32 bytes *)
  let wide =
    one_block [ Mir_type.Ptr; i64 ] [ i32 ] (fun bld e -> function
      | [ base; off ] ->
          let addr = B.emit bld e (Mir_op.Ptr_add (base, off)) in
          B.emit_unit bld e
            (Mir_op.Store
               ( { Mir_op.Access.width = Mir_width.W32; addr; align = 4L },
                 k32 bld e 0x5EEDL ));
          [ load32 bld e addr ]
      | _ -> assert false)
    |> with_objects
         ~regions:
           [
             {
               (region 0 0x2_0000_0000L) with
               Mir_region.init = Mir_region.Uninitialized;
             };
           ]
         ~views:[ view 0 ~region:0 0x2_0000_0000L ]
  in
  print_run (run wide [ Ptr 0; i64v 0x1_0000_0008L ]);
  print_run (run wide [ Ptr 0; i64v 0x1_FFFF_FFFEL ]);
  [%expect
    {|
    success [0x40400000]
    defect bad_access at fn0 bb0 i4
    defect bad_access at fn0 bb0 i4
    defect bad_access at fn0 bb0 i4
    success [0x5eed]
    defect bad_access at fn0 bb0 i2 |}]

let%expect_test "overlapping views and partial initialization" =
  (* view 0 and view 1 overlap: a store through one is read through the other *)
  let alias =
    one_block [ Mir_type.Ptr; Mir_type.Ptr ] [ i32 ] (fun bld e -> function
      | [ a; b ] ->
          let a4 = B.emit bld e (Mir_op.Ptr_add (a, k64 bld e 4L)) in
          B.emit_unit bld e
            (Mir_op.Store
               ( { Mir_op.Access.width = Mir_width.W32; addr = a4; align = 4L },
                 k32 bld e 42L ));
          [ load32 bld e b ]
      | _ -> assert false)
    |> with_objects
         ~regions:[ region 0 12L ]
         ~views:[ view 0 ~region:0 12L; view 1 ~region:0 ~offset:4L 8L ]
  in
  print_run (run alias [ Ptr 0; Ptr 1 ]);
  (* one byte written into an otherwise undefined cell *)
  let partial =
    one_block [ Mir_type.Ptr ] [ i32 ] (fun bld e -> function
      | [ a ] ->
          B.emit_unit bld e
            (Mir_op.Store
               ( { Mir_op.Access.width = Mir_width.W8; addr = a; align = 1L },
                 const bld e (Mir_const.int Mir_width.W8 7L) ));
          [ load32 bld e a ]
      | _ -> assert false)
    |> with_objects ~regions:[ region 0 4L ] ~views:[ view 0 ~region:0 4L ]
  in
  print_run (run partial [ Ptr 0 ]);
  [%expect {|
    success [0x2a]
    defect uninitialized at fn0 bb0 i2 |}]

(* Stores [x] as binary32 and reads back the four bytes. [truncate] is the
   store-rounding mutation: a value that rounded away from zero steps back one
   unit (toward zero), for a positive [x]. [event] drops the event when false. *)
let stored_bits ?(truncate = false) ?(event = true) x =
  let p =
    one_block [ Mir_type.Ptr; Mir_type.F64 ] [] (fun bld e -> function
      | [ base; x ] ->
          let r =
            B.emit bld e (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, x))
          in
          let bits = B.emit bld e (Mir_op.Bitcast (Mir_type.i32, r)) in
          let bits =
            if not truncate then bits
            else
              let back =
                B.emit bld e (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, r))
              in
              let up = B.emit bld e (Mir_op.Fcmp (Mir_op.Fcmp.Lt, x, back)) in
              let less =
                B.emit bld e
                  (Mir_op.Iarith (Mir_op.Iarith.Sub, bits, k32 bld e 1L))
              in
              B.emit bld e (Mir_op.Select (up, less, bits))
          in
          B.emit_unit bld e
            (Mir_op.Store
               ( { Mir_op.Access.width = Mir_width.W32; addr = base; align = 4L },
                 bits ));
          if event then B.emit_unit bld e (Mir_op.Event (Mir_event.Emitter, 1L));
          []
      | _ -> assert false)
    |> with_objects
         ~regions:[ region 0 4L ]
         ~views:[ view 0 ~region:0 ~role:Mir_view.Output 4L ]
  in
  let r, memory, binding = run p [ Ptr 0; Val (Mir_datum.f64 x) ] in
  let key =
    Option.get (Mir_interp.Binding.instance binding (Mir_id.Region.of_int 0))
  in
  let bytes = Mir_memory.read_bytes memory key ~offset:0L ~n:4 in
  ( show_outcome r,
    Array.fold_right
      (fun b acc ->
        match b with
        | Some b -> Printf.sprintf "%02x" b ^ acc
        | None -> "??" ^ acc)
      bytes "" )

let%expect_test "exact storage bits, store rounding and events" =
  let show (o, b) = Fmt.pr "%s bytes %s@." o b in
  show (stored_bits 0.1);
  show (stored_bits (-0.));
  (* above binary32's largest finite value by more than half a unit *)
  show (stored_bits 3.4028235677973366e38);
  (* the mutations: a truncating store and a dropped event *)
  show (stored_bits ~truncate:true 0.1);
  show (stored_bits ~event:false 0.1);
  [%expect
    {|
    success [] events emitter=1 bytes cdcccc3d
    success [] events emitter=1 bytes 00000080
    success [] events emitter=1 bytes 0000807f
    success [] events emitter=1 bytes cccccc3d
    success [] bytes cdcccc3d |}]

let%expect_test "checked division: failure selection and order" =
  let d x y =
    show_outcome
      (let r, _, _ = run (div ()) [ i64v x; i64v y ] in
       r)
  in
  List.iter print_endline
    [
      d 7L (-2L);
      d 7L 0L;
      d Int64.min_int (-1L);
      d Int64.min_int 0L;
      d Int64.min_int 1L;
    ];
  [%expect
    {|
    success [0xfffffffffffffffd]
    failure i64_division_by_zero()
    failure i64_division_overflow()
    failure i64_division_by_zero()
    success [0x8000000000000000] |}]
