(* What the Coral route plans for a computation: the matrix-multiply engine
   chosen for each contraction, then the final SSA program of each invocation,
   exactly the one lowered. Planning only: nothing is assembled or simulated,
   so a change to selection, blocking, numerics or fusion shows here as a
   readable diff before any RTL or MPACT run. *)

module Cases = Coral_estimate_cases
module M = Machine_model.Mir_model
module P = Ssa_ir.Ssa_contract_plan

let capabilities ~matrix =
  { Contraction.Select.Capabilities.matrix; vector = true }

let case name =
  match List.find_opt (fun (c : Cases.t) -> c.Cases.name = name) Cases.all with
  | Some c -> c
  | None -> Fmt.failwith "no case %S" name

let show ~matrix name =
  let c = case name in
  let b =
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
      (Loop_ir.Loop_bundle.build c.Cases.graph)
  in
  let route =
    let module R = Machine_model.Mir_model_route in
    R.Route.Custom
      {
        R.Custom.name = "planned SSA, generic execution";
        plan =
          (fun p ->
            Err.payload (P.program ~capabilities:(capabilities ~matrix) p)
            |> Result.map_error (Fmt.str "%a" P.pp_error));
        exec = (fun ~sites:_ g -> Ok (R.generic g));
      }
  in
  let n = ref 0 in
  let on_program _ p _ =
    incr n;
    Fmt.pr "-- invocation %d@." !n;
    List.iter
      (fun (_, e) ->
        Fmt.pr "contraction engine: %s@."
          (match e with
          | Contraction.Select.Engine.Loops -> "loops"
          | Rvv -> "rvv"
          | Zvt -> "zvt"))
      (P.choices p);
    Fmt.pr "%s@." (Ssa_ir.Ssa_pp.to_string p)
  in
  Fmt.pr "== %s, matrix engine %s@." name (if matrix then "on" else "off");
  match
    M.prepare ~route ~contractions:c.Cases.contractions
      ~bf16_pairs:c.Cases.bf16_pairs ~pipeline:(Cases.pipeline c) ~on_program b
  with
  | Error rs ->
      Fmt.pr "refused: %a@." Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs
  | Ok _ -> ()

let%expect_test "plans" =
  show ~matrix:true "add 16x64";
  show ~matrix:true "add 16x64, scalar";
  show ~matrix:true "exp 16x16";
  show ~matrix:true "x*x+x 16x64";
  show ~matrix:true "linear bf16 9x32 k64";
  show ~matrix:false "linear bf16 9x32 k64";
  show ~matrix:true "linear 9x32 k64";
  show ~matrix:false "linear 9x32 k64";
  [%expect
    {|
    == add 16x64, matrix engine on
    -- invocation 1
    buffer b0 in f32 [1, 1, 1, 1, 1, 1024]
    buffer b1 in f32 [1, 1, 1, 1, 1, 1024]
    buffer b2 out f32 [1, 1, 1, 1, 1, 1024]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 1024:index
      (%3:index) = const 16:index
      (%4:effect) = for %1 to %2 step 32 iter(%5:index, %6:effect := %0) {
        (%7:index) = index.add_in_domain %5, %3
        (%8:vec<x16,f32>, %9:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %6
        (%10:vec<x16,f32>, %11:effect) = vec.load.f32_exact b1[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %9
        (%12:vec<x16,f32>, %13:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %11
        (%14:vec<x16,f32>) = lanes float.add %8, %10
        (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b1[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %13
        (%17:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16, %14 effect %16
        (%18:vec<x16,f32>) = lanes float.add %12, %15
        (%19:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16, %18 effect %17
        yield %19
      }
      yield %4
    }

    == add 16x64, scalar, matrix engine on
    -- invocation 1
    buffer b0 in f32 [1, 1, 1, 1, 16, 64]
    buffer b1 in f32 [1, 1, 1, 1, 16, 64]
    buffer b2 out f32 [1, 1, 1, 1, 16, 64]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 16:index
      (%3:index) = const 64:index
      (%4:effect) = for %1 to %2 step 1 iter(%5:index, %6:effect := %0) {
        (%7:effect) = for %1 to %3 step 1 iter(%8:index, %9:effect := %6) {
          (%10:f32, %11:effect) = load.in_bounds.f32_exact b0[%1, %1, %1, %1, %5, %8] effect %9
          (%12:f32, %13:effect) = load.in_bounds.f32_exact b1[%1, %1, %1, %1, %5, %8] effect %11
          (%14:f32) = float.add %10, %12
          (%15:effect) = store.f32_exact b2[%1, %1, %1, %1, %5, %8], %14 effect %13
          yield %15
        }
        yield %7
      }
      yield %4
    }

    == exp 16x16, matrix engine on
    -- invocation 1
    buffer b0 in f32 [1, 1, 1, 1, 1, 256]
    buffer b1 out f32 [1, 1, 1, 1, 1, 256]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 256:index
      (%3:effect) = for %1 to %2 step 16 iter(%4:index, %5:effect := %0) {
        (%6:vec<x16,f32>, %7:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %4] step [0, 0, 0, 0, 0, 1] x16 effect %5
        (%8:vec<x16,f32>) = lanes float.exp %6
        (%9:effect) = vec.store.f32_exact b1[%1, %1, %1, %1, %1, %4] step [0, 0, 0, 0, 0, 1] x16, %8 effect %7
        yield %9
      }
      yield %3
    }

    == x*x+x 16x64, matrix engine on
    -- invocation 1
    buffer b0 in f32 [1, 1, 1, 1, 1, 1024]
    buffer b1 out f32 [1, 1, 1, 1, 1, 1024]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 1024:index
      (%3:index) = const 16:index
      (%4:effect) = for %1 to %2 step 32 iter(%5:index, %6:effect := %0) {
        (%7:index) = index.add_in_domain %5, %3
        (%8:vec<x16,f32>, %9:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %6
        (%10:vec<x16,f32>, %11:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %9
        (%12:vec<x16,f32>, %13:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %11
        (%14:vec<x16,f32>) = lanes float.mul %8, %10
        (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %13
        (%17:effect) = vec.store.f32_exact b1[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16, %14 effect %16
        (%18:vec<x16,f32>) = lanes float.mul %12, %15
        (%19:effect) = vec.store.f32_exact b1[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16, %18 effect %17
        yield %19
      }
      yield %4
    }

    -- invocation 2
    buffer b1 in f32 [1, 1, 1, 1, 1, 1024]
    buffer b0 in f32 [1, 1, 1, 1, 1, 1024]
    buffer b2 out f32 [1, 1, 1, 1, 1, 1024]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 1024:index
      (%3:index) = const 16:index
      (%4:effect) = for %1 to %2 step 32 iter(%5:index, %6:effect := %0) {
        (%7:index) = index.add_in_domain %5, %3
        (%8:vec<x16,f32>, %9:effect) = vec.load.f32_exact b1[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %6
        (%10:vec<x16,f32>, %11:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16 effect %9
        (%12:vec<x16,f32>, %13:effect) = vec.load.f32_exact b1[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %11
        (%14:vec<x16,f32>) = lanes float.add %8, %10
        (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b0[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16 effect %13
        (%17:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %1, %5] step [0, 0, 0, 0, 0, 1] x16, %14 effect %16
        (%18:vec<x16,f32>) = lanes float.add %12, %15
        (%19:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %1, %7] step [0, 0, 0, 0, 0, 1] x16, %18 effect %17
        yield %19
      }
      yield %4
    }

    == linear bf16 9x32 k64, matrix engine on
    -- invocation 1
    contraction engine: zvt
    buffer b0 in bf16 [1, 1, 1, 1, 9, 64]
    buffer b1 in bf16 [32, 1, 1, 1, 1, 64]
    buffer b3 scratch f32 [1, 1, 1, 1, 1, 32]
    buffer b2 out f32 [1, 1, 1, 1, 9, 32]
    buffer b4 scratch f32 [1, 1, 1, 1, 9, 32]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 9:index
      (%3:index) = const 32:index
      (%4:f32) = const 0x0p+0:f32
      (%5:effect) = contract bf16 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero paired engine=zvt a=b0 b=b1 c=b4 effect %0
      (%6:index) = const 16:index
      (%7:vec<x16,f32>) = vec.splat x16 %4
      (%8:effect) = for %1 to %2 step 1 iter(%9:index, %10:effect := %5) {
        (%11:effect) = for %1 to %3 step 32 iter(%12:index, %13:effect := %10) {
          (%14:index) = index.add_in_domain %12, %6
          (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16 effect %13
          (%17:vec<x16,f32>) = lanes float.add %15, %7
          (%18:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16, %17 effect %16
          (%19:vec<x16,f32>, %20:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16 effect %18
          (%21:vec<x16,f32>) = lanes float.add %19, %7
          (%22:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16, %21 effect %20
          yield %22
        }
        yield %11
      }
      yield %8
    }

    == linear bf16 9x32 k64, matrix engine off
    refused: invocation 0 (n0): loops cannot realize contract.bf16.b1.m9.n32.k64.a0_64_1.b0_64_1.c288_32_1.zero.paired (matrix=false, vector=true)
    == linear 9x32 k64, matrix engine on
    -- invocation 1
    contraction engine: zvt
    buffer b0 in f32 [1, 1, 1, 1, 9, 64]
    buffer b1 in f32 [32, 1, 1, 1, 1, 64]
    buffer b3 scratch f32 [1, 1, 1, 1, 1, 32]
    buffer b2 out f32 [1, 1, 1, 1, 9, 32]
    buffer b4 scratch f32 [1, 1, 1, 1, 9, 32]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 9:index
      (%3:index) = const 32:index
      (%4:f32) = const 0x0p+0:f32
      (%5:effect) = contract f32 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero ordered engine=zvt a=b0 b=b1 c=b4 effect %0
      (%6:index) = const 16:index
      (%7:vec<x16,f32>) = vec.splat x16 %4
      (%8:effect) = for %1 to %2 step 1 iter(%9:index, %10:effect := %5) {
        (%11:effect) = for %1 to %3 step 32 iter(%12:index, %13:effect := %10) {
          (%14:index) = index.add_in_domain %12, %6
          (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16 effect %13
          (%17:vec<x16,f32>) = lanes float.add %15, %7
          (%18:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16, %17 effect %16
          (%19:vec<x16,f32>, %20:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16 effect %18
          (%21:vec<x16,f32>) = lanes float.add %19, %7
          (%22:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16, %21 effect %20
          yield %22
        }
        yield %11
      }
      yield %8
    }

    == linear 9x32 k64, matrix engine off
    -- invocation 1
    contraction engine: rvv
    buffer b0 in f32 [1, 1, 1, 1, 9, 64]
    buffer b1 in f32 [32, 1, 1, 1, 1, 64]
    buffer b3 scratch f32 [1, 1, 1, 1, 1, 32]
    buffer b2 out f32 [1, 1, 1, 1, 9, 32]
    buffer b4 scratch f32 [1, 1, 1, 1, 9, 32]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 9:index
      (%3:index) = const 32:index
      (%4:f32) = const 0x0p+0:f32
      (%5:effect) = contract f32 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero ordered engine=rvv a=b0 b=b1 c=b4 effect %0
      (%6:index) = const 16:index
      (%7:vec<x16,f32>) = vec.splat x16 %4
      (%8:effect) = for %1 to %2 step 1 iter(%9:index, %10:effect := %5) {
        (%11:effect) = for %1 to %3 step 32 iter(%12:index, %13:effect := %10) {
          (%14:index) = index.add_in_domain %12, %6
          (%15:vec<x16,f32>, %16:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16 effect %13
          (%17:vec<x16,f32>) = lanes float.add %15, %7
          (%18:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %12] step [0, 0, 0, 0, 0, 1] x16, %17 effect %16
          (%19:vec<x16,f32>, %20:effect) = vec.load.f32_exact b4[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16 effect %18
          (%21:vec<x16,f32>) = lanes float.add %19, %7
          (%22:effect) = vec.store.f32_exact b2[%1, %1, %1, %1, %9, %14] step [0, 0, 0, 0, 0, 1] x16, %21 effect %20
          yield %22
        }
        yield %11
      }
      yield %8
    } |}]
