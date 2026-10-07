open Machine_ir
open Machine_interp
open Machine_target_aarch64
module B = Mir_builder
module H = Alloc_harness

(* A vector live across a call on AArch64: AAPCS64 preserves only the low 64
   bits of v8-v15, so a Q value there does not survive the call. Linear scan
   given only v8-v10 must keep the vector elsewhere — its slot — and with the
   call's clobbers ignored the checker catches the vector left in v8. *)

let f64 = Mir_type.F64
let v4 = Mir_type.Vec (Mir_type.Elem.F32, Mir_type.Lanes.of_int 4)
let fn0 = Mir_id.Func.of_int 0

let program =
  let bld = B.create () in
  let e = B.new_block bld [ f64 ] in
  let x = List.hd (B.param e) in
  let access view =
    {
      Mir_op.Vaccess.elem = Mir_type.Elem.F32;
      lanes = Mir_type.Lanes.of_int 4;
      addr = B.emit bld e (Mir_op.Addr (Mir_id.View.of_int view));
      stride = 4L;
      align = 4L;
    }
  in
  let v = B.emit bld e (Mir_op.Vload (access 0)) in
  let _ =
    Result.get_ok
      (B.op bld e
         ~signature:(fun _ ->
           Some { Mir_typing.Signature.params = [ f64 ]; results = [ f64 ] })
         (Mir_op.Call
            (Mir_op.Callee.Helper Calls_test.scale.Mir_helper.id, [ x ])))
  in
  let w = B.emit bld e (Mir_op.Fbinary (Mir_op.Fbinary.Add, v, v)) in
  B.emit_unit bld e (Mir_op.Vstore (access 1, w));
  B.return e [];
  let main = B.func bld ~id:fn0 ~name:"main" ~entry:e ~results:[] in
  let region k init =
    { Mir_region.id = Mir_id.Region.of_int k; size = 16L; align = 16L; init }
  in
  let view k perm role =
    {
      Mir_view.id = Mir_id.View.of_int k;
      region = Mir_id.Region.of_int k;
      offset = 0L;
      size = 16L;
      perm;
      role;
      source = None;
    }
  in
  B.program
    ~regions:[ region 0 Mir_region.Bound; region 1 Mir_region.Bound ]
    ~views:
      [
        view 0 Mir_view.Read Mir_view.Input;
        view 1 Mir_view.Read_write Mir_view.Output;
      ]
    ~helpers:[ Calls_test.scale ] [ main ] ~main:fn0

module Pool = struct
  include A64_regs

  (* only registers whose upper half a call clobbers *)
  let allocatable = function
    | Mir_target.Bank.Fpr -> [ 8; 9; 10 ]
    | bank -> allocatable bank
end

module Ls = Machine_alloc.Mir_linear_scan.Make (A64) (Pool)

let bytes =
  let b = Bytes.create 16 in
  List.iteri
    (fun k x -> Bytes.set_int32_le b (4 * k) (Int32.bits_of_float x))
    [ 1.5; -0.; 3.25; 1e30 ];
  Bytes.to_string b

let bound r =
  if Mir_id.Region.equal r (Mir_id.Region.of_int 0) then Some bytes
  else Some (String.make 16 '\000')

let out memory binding =
  let key =
    Option.get (Mir_interp.Binding.instance binding (Mir_id.Region.of_int 1))
  in
  let base = Mir_memory.pointer memory key ~lo:0L ~hi:16L in
  String.concat " "
    (List.init 4 (fun k ->
         match
           Mir_memory.load memory
             (Option.get (Mir_memory.offset_by base (Int64.of_int (4 * k))))
             ~bytes:4L ~align:1L
         with
         | Ok b -> Fmt.str "%h" (Int32.float_of_bits (Int64.to_int32 b))
         | Error _ -> "undefined"))

let run ?mutation () =
  let g = Result.get_ok (Err.payload (Mir_verify.generic program)) in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let phys = Ls.allocate ?mutation res.A64_select.selected in
  let slots = (Machine_alloc.Mir_alloc_stats.of_program phys).slot_bytes in
  match Err.payload (H.C.check res.A64_select.selected phys) with
  | Error e -> Fmt.pr "checker: %a@." Machine_check.Mir_checker.pp_error e
  | Ok () ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate (H.regions_program phys) memory ~bound)
      in
      let r =
        H.P.run ~models:Calls_test.models phys memory binding
          ~args:[ Mir_datum.f64 2. ]
      in
      Fmt.pr "%a; %Ld slot bytes; out %s@." Mir_interp.Outcome.pp r.H.P.outcome
        slots (out memory binding)

let%expect_test "a vector live across a call" =
  run ();
  run ~mutation:Machine_alloc.Mir_linear_scan.Mutation.Call_interval ();
  [%expect
    {|
    success [0x0]; 16 slot bytes; out 0x1.8p+1 -0x0p+0 0x1.ap+2 0x1.93e594p+100
    checker: fn0 bb0: q9 does not hold %10 |}]
