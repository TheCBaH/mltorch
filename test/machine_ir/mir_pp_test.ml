open Machine_ir
open Mir_fixtures

(* The printer: canonical text, independent of the ids a builder happened to
   allocate and of the order blocks are listed after the entry. *)

let text p = Fmt.str "%a" (Mir_pp.generic ~origins:false) p

(* Every value and block id moved and the non-entry blocks reversed: the same
   program under a different allocation history. *)
let renumber (p : Mir_program.generic) =
  let v (x : Mir_value.t) =
    {
      x with
      Mir_value.id =
        Mir_id.Value.of_int (5000 - Mir_id.Value.to_int x.Mir_value.id);
    }
  in
  let b x = Mir_id.Block.of_int (77 + (3 * Mir_id.Block.to_int x)) in
  let edge (e : Mir_edge.t) =
    {
      Mir_edge.target = b e.Mir_edge.target;
      args = List.map v e.Mir_edge.args;
      order = v e.Mir_edge.order;
    }
  in
  let term = function
    | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
        Mir_terminator.Branch
          { Mir_branch.cond = v cond; then_ = edge then_; else_ = edge else_ }
    | Mir_terminator.Fail f ->
        Mir_terminator.Fail
          {
            f with
            Mir_fail.payload = List.map v f.Mir_fail.payload;
            order = v f.Mir_fail.order;
          }
    | Mir_terminator.Jump e -> Mir_terminator.Jump (edge e)
    | Mir_terminator.Return { Mir_return.values; order } ->
        Mir_terminator.Return
          { Mir_return.values = List.map v values; order = v order }
  in
  let instr (i : Mir_op.t Mir_instr.t) =
    {
      i with
      Mir_instr.id =
        Mir_id.Instr.of_int (900 + Mir_id.Instr.to_int i.Mir_instr.id);
      results = List.map v i.Mir_instr.results;
      op = Mir_op.map_operands v i.Mir_instr.op;
      order =
        Option.map
          (fun o ->
            {
              Mir_order.input = v o.Mir_order.input;
              output = v o.Mir_order.output;
            })
          i.Mir_instr.order;
    }
  in
  let block (bl : (_, _) Mir_block.t) =
    {
      Mir_block.id = b bl.Mir_block.id;
      params = List.map v bl.Mir_block.params;
      order = v bl.Mir_block.order;
      body = List.map instr bl.Mir_block.body;
      terminator = term bl.Mir_block.terminator;
    }
  in
  {
    p with
    Mir_program.revision = Mir_id.Revision.of_int 9;
    funcs =
      List.map
        (fun (f : (_, _) Mir_func.t) ->
          match List.map block f.Mir_func.blocks with
          | e :: rest ->
              {
                f with
                Mir_func.entry = b f.Mir_func.entry;
                blocks = e :: List.rev rest;
              }
          | [] -> f)
        p.Mir_program.funcs;
  }

let%expect_test "canonical text" =
  print_endline (text (sum_f32 ()));
  [%expect
    {|
    data_model lp64-le-elf
    planning none
    fn0 main func sum_f32 -> (f64) {
    bb0(%0: ptr64, %1: i32; !2):
      %3 = const 0:i32
      %4 = const 0x0p+0:f64
      jump bb1(%3, %4; !2)
    bb1(%5: i32, %6: f64; !7):
      %8 = icmp.slt %5, %1
      branch %8, bb3(; !7), bb2(%6; !7)
    bb2(%9: f64; !10):
      return %9; !10
    bb3(; !11):
      %12 = const 2147483647:i32
      %13 = const 0:i32
      %14 = icmp.sle %13, %5
      %15 = icmp.slt %5, %12
      %16 = pand %14, %15
      branch %16, bb5(; !11), bb4(; !11)
    bb4(; !17):
      %18 = sext.i64 %5
      %19 = const 0:i64
      fail coord_out_of_range(t7, W)(%19, %19, %19, %19, %18, %19; !17)
    bb5(; !20):
      %21 = sext.i64 %5
      %22 = const 4:i64
      %23 = mul %21, %22
      %24 = ptr.add %0, %23
      %25, !26 = load.i32 [%24] align 4, !20
      %27 = bitcast.f32 %25
      %28 = fext.f32.f64 %27
      %29 = fadd %6, %28
      %30 = const 1:i32
      %31 = add %5, %30
      jump bb1(%31, %29; !26)
    } |}]

let%expect_test "allocation history does not change the text" =
  List.iter
    (fun p ->
      let q = renumber p in
      (match Err.payload (Mir_verify.generic q) with
      | Ok _ -> ()
      | Error d -> Fmt.pr "renumbered: %a@." Mir_diagnostic.pp d);
      Fmt.pr "%b@." (String.equal (text p) (text q)))
    [ swap (); sum_f32 (); div (); store_f32 () ];
  [%expect {|
    true
    true
    true
    true |}]
