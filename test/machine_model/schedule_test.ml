(* M10.2: block-local scheduling of the model kernels' selected programs on
   both targets — every schedule within its source order's dependences and
   reverified, what it moves, what linear scan then costs, and each dropped
   dependence class caught. *)

open Machine_ir

module Run
    (T : Mir_sel.TARGET)
    (R : Machine_alloc.Mir_linear_scan.POOL)
    (X : sig
      val name : string

      val select :
        sites:Mir_failure.Site_entry.t array ->
        Mir_verify.Generic.t ->
        Mir_sel.Make(T).Verified.t
    end) =
struct
  module Sch = Machine_alloc.Mir_schedule.Make (T)
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module Policy = Machine_alloc.Mir_schedule.Policy

  let moved (before : Sch.S.Verified.t) (after : Sch.S.Verified.t) =
    let bodies v =
      List.concat_map
        (fun (f : (_, _) Mir_func.t) ->
          List.map
            (fun (b : (_, _) Mir_block.t) ->
              List.map
                (fun (i : _ Mir_instr.t) -> i.Mir_instr.id)
                b.Mir_block.body)
            f.Mir_func.blocks)
        (Sch.S.Verified.selected v).Sch.S.program.Mir_program.funcs
    in
    List.fold_left2
      (fun n a b ->
        List.fold_left2 (fun n x y -> if x <> y then n + 1 else n) n a b)
      0 (bodies before) (bodies after)

  let compare () =
    List.iter
      (fun (name, gen, sites) ->
        let v = X.select ~sites gen in
        match (Sch.schedule Policy.Sink v, Sch.schedule Policy.Reverse v) with
        | Error r, _ | _, Error r ->
            Fmt.pr "%s %s: %a@." X.name name
              Machine_alloc.Mir_schedule.Refusal.pp r
        | Ok s, Ok r ->
            Fmt.pr
              "%s %s: sink moves %d (%a), reverse %d@.  source: %a@.  sink: \
               %a@."
              X.name name (moved v s) Mir_id.Revision.pp
              (Sch.S.Verified.selected s).Sch.S.program.Mir_program.revision
              (moved v r) Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate v))
              Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate s)))
      (Scan_test.kernels ())

  (* Each class dropped from the graph under the reverse order: the check
     refuses, and how often the selected verifier alone rejects the same order.
     The graph keeps a flags writer on the side of a condition range it started
     on, which is stricter than the verifier's rule: on AArch64 every writer the
     reverse order moves lands outside any range, so only the check sees it. *)
  let mutations () =
    List.iter
      (fun (label, mutation) ->
        let caught =
          List.filter_map
            (fun (name, gen, sites) ->
              let v = X.select ~sites gen in
              match Sch.schedule ~mutation Policy.Reverse v with
              | Ok _ -> None
              | Error r ->
                  let verifier =
                    Result.is_error
                      (Err.payload
                         (Sch.S.Verified.verify
                            (Sch.reorder ~mutation Policy.Reverse v)))
                  in
                  Some (name, r, verifier))
            (Scan_test.kernels ())
        in
        match caught with
        | [] -> Fmt.pr "%s %s: NOT CAUGHT@." X.name label
        | (name, r, _) :: _ ->
            Fmt.pr
              "%s %s: caught on %d kernels (verifier alone: %d); first %s: %a@."
              X.name label (List.length caught)
              (List.length (List.filter (fun (_, _, v) -> v) caught))
              name Machine_alloc.Mir_schedule.Refusal.pp r)
      Machine_alloc.Mir_schedule.Mutation.
        [ ("data", Data); ("flags", Flags); ("order", Order) ]
end

module A64 =
  Run (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64"
    end)

module X64 =
  Run (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64"
    end)

module A64_small =
  Run
    (Machine_target_aarch64.A64)
    (struct
      include Machine_target_aarch64.A64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64, 3 registers"
    end)

module X64_small =
  Run
    (Machine_target_x86_64.X64)
    (struct
      include Machine_target_x86_64.X64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64, 3 registers"
    end)

let%expect_test "dropped dependences are caught" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i230 scheduled after i238, which depends on it
    aarch64 flags: caught on 4 kernels (verifier alone: 0); first chain/0: dependence: bb4: i256 scheduled after i262, which depends on it
    aarch64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i371 scheduled after i372, which depends on it
    x86_64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i230 scheduled after i237, which depends on it
    x86_64 flags: caught on 4 kernels (verifier alone: 2); first chain/0: dependence: bb4: i254 scheduled after i260, which depends on it
    x86_64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i380 scheduled after i381, which depends on it |}]

let%expect_test "sink scheduling, then linear scan" =
  A64.compare ();
  X64.compare ();
  A64_small.compare ();
  X64_small.compare ();
  [%expect
    {|
    aarch64 chain/0: sink moves 8 (rev1), reverse 97
      source: 111 instructions; 34 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
      sink: 111 instructions; 34 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1: sink moves 0 (rev0), reverse 57
      source: 68 instructions; 16 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 68 instructions; 16 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2: sink moves 0 (rev0), reverse 27
      source: 41 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 41 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0: sink moves 0 (rev0), reverse 55
      source: 65 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 65 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0: sink moves 13 (rev1), reverse 181
      source: 205 instructions; 54 register moves, 6 stores, 6 loads, 0 slot moves, 0 rematerialized; 6 slots (40 bytes)
      sink: 205 instructions; 53 register moves, 5 stores, 5 loads, 0 slot moves, 0 rematerialized; 5 slots (36 bytes)
    aarch64 sdpa/0: sink moves 15 (rev1), reverse 498
      source: 559 instructions; 101 register moves, 30 stores, 27 loads, 0 slot moves, 0 rematerialized; 14 slots (100 bytes)
      sink: 559 instructions; 99 register moves, 26 stores, 23 loads, 0 slot moves, 0 rematerialized; 14 slots (100 bytes)
    aarch64 conv/0: sink moves 8 (rev1), reverse 113
      source: 132 instructions; 39 register moves, 3 stores, 3 loads, 0 slot moves, 3 rematerialized; 3 slots (24 bytes)
      sink: 132 instructions; 39 register moves, 3 stores, 3 loads, 0 slot moves, 3 rematerialized; 3 slots (24 bytes)
    x86_64 chain/0: sink moves 14 (rev1), reverse 107
      source: 116 instructions; 52 register moves, 18 stores, 16 loads, 0 slot moves, 3 rematerialized; 13 slots (84 bytes)
      sink: 116 instructions; 52 register moves, 18 stores, 16 loads, 0 slot moves, 3 rematerialized; 13 slots (84 bytes)
    x86_64 chain/1: sink moves 6 (rev1), reverse 67
      source: 73 instructions; 25 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
      sink: 73 instructions; 23 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
    x86_64 chain/2: sink moves 6 (rev1), reverse 35
      source: 45 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
      sink: 45 instructions; 24 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0: sink moves 9 (rev1), reverse 59
      source: 65 instructions; 31 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
      sink: 65 instructions; 30 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
    x86_64 softmax/0: sink moves 14 (rev1), reverse 216
      source: 230 instructions; 58 register moves, 23 stores, 23 loads, 0 slot moves, 2 rematerialized; 12 slots (84 bytes)
      sink: 230 instructions; 58 register moves, 23 stores, 23 loads, 0 slot moves, 2 rematerialized; 12 slots (84 bytes)
    x86_64 sdpa/0: sink moves 27 (rev1), reverse 553
      source: 601 instructions; 93 register moves, 50 stores, 49 loads, 0 slot moves, 13 rematerialized; 20 slots (148 bytes)
      sink: 601 instructions; 93 register moves, 50 stores, 49 loads, 0 slot moves, 13 rematerialized; 20 slots (148 bytes)
    x86_64 conv/0: sink moves 14 (rev1), reverse 119
      source: 130 instructions; 59 register moves, 21 stores, 19 loads, 0 slot moves, 3 rematerialized; 16 slots (104 bytes)
      sink: 130 instructions; 59 register moves, 21 stores, 19 loads, 0 slot moves, 3 rematerialized; 16 slots (104 bytes)
    aarch64, 3 registers chain/0: sink moves 8 (rev1), reverse 97
      source: 111 instructions; 28 register moves, 67 stores, 53 loads, 0 slot moves, 9 rematerialized; 23 slots (148 bytes)
      sink: 111 instructions; 27 register moves, 66 stores, 53 loads, 0 slot moves, 10 rematerialized; 23 slots (148 bytes)
    aarch64, 3 registers chain/1: sink moves 0 (rev0), reverse 57
      source: 68 instructions; 16 register moves, 20 stores, 17 loads, 0 slot moves, 5 rematerialized; 9 slots (60 bytes)
      sink: 68 instructions; 16 register moves, 20 stores, 17 loads, 0 slot moves, 5 rematerialized; 9 slots (60 bytes)
    aarch64, 3 registers chain/2: sink moves 0 (rev0), reverse 27
      source: 41 instructions; 8 register moves, 13 stores, 11 loads, 0 slot moves, 5 rematerialized; 7 slots (44 bytes)
      sink: 41 instructions; 8 register moves, 13 stores, 11 loads, 0 slot moves, 5 rematerialized; 7 slots (44 bytes)
    aarch64, 3 registers bmm/0: sink moves 0 (rev0), reverse 55
      source: 65 instructions; 14 register moves, 27 stores, 24 loads, 0 slot moves, 7 rematerialized; 12 slots (80 bytes)
      sink: 65 instructions; 14 register moves, 27 stores, 24 loads, 0 slot moves, 7 rematerialized; 12 slots (80 bytes)
    aarch64, 3 registers softmax/0: sink moves 13 (rev1), reverse 181
      source: 205 instructions; 19 register moves, 48 stores, 41 loads, 0 slot moves, 14 rematerialized; 17 slots (120 bytes)
      sink: 205 instructions; 19 register moves, 46 stores, 39 loads, 0 slot moves, 15 rematerialized; 16 slots (112 bytes)
    aarch64, 3 registers sdpa/0: sink moves 15 (rev1), reverse 498
      source: 559 instructions; 45 register moves, 129 stores, 108 loads, 0 slot moves, 24 rematerialized; 29 slots (212 bytes)
      sink: 559 instructions; 42 register moves, 127 stores, 106 loads, 0 slot moves, 25 rematerialized; 29 slots (212 bytes)
    aarch64, 3 registers conv/0: sink moves 8 (rev1), reverse 113
      source: 132 instructions; 21 register moves, 73 stores, 59 loads, 0 slot moves, 14 rematerialized; 26 slots (168 bytes)
      sink: 132 instructions; 21 register moves, 72 stores, 58 loads, 0 slot moves, 15 rematerialized; 26 slots (168 bytes)
    x86_64, 3 registers chain/0: sink moves 14 (rev1), reverse 107
      source: 116 instructions; 24 register moves, 61 stores, 61 loads, 0 slot moves, 14 rematerialized; 23 slots (148 bytes)
      sink: 116 instructions; 25 register moves, 60 stores, 60 loads, 0 slot moves, 15 rematerialized; 23 slots (148 bytes)
    x86_64, 3 registers chain/1: sink moves 6 (rev1), reverse 67
      source: 73 instructions; 11 register moves, 22 stores, 23 loads, 0 slot moves, 7 rematerialized; 11 slots (76 bytes)
      sink: 73 instructions; 10 register moves, 21 stores, 22 loads, 0 slot moves, 7 rematerialized; 11 slots (76 bytes)
    x86_64, 3 registers chain/2: sink moves 6 (rev1), reverse 35
      source: 45 instructions; 13 register moves, 16 stores, 18 loads, 0 slot moves, 7 rematerialized; 7 slots (44 bytes)
      sink: 45 instructions; 12 register moves, 15 stores, 17 loads, 0 slot moves, 7 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers bmm/0: sink moves 9 (rev1), reverse 59
      source: 65 instructions; 17 register moves, 29 stores, 32 loads, 0 slot moves, 10 rematerialized; 12 slots (80 bytes)
      sink: 65 instructions; 17 register moves, 28 stores, 31 loads, 0 slot moves, 10 rematerialized; 12 slots (80 bytes)
    x86_64, 3 registers softmax/0: sink moves 14 (rev1), reverse 216
      source: 230 instructions; 27 register moves, 41 stores, 48 loads, 0 slot moves, 28 rematerialized; 15 slots (104 bytes)
      sink: 230 instructions; 29 register moves, 42 stores, 50 loads, 0 slot moves, 28 rematerialized; 14 slots (96 bytes)
    x86_64, 3 registers sdpa/0: sink moves 27 (rev1), reverse 553
      source: 601 instructions; 44 register moves, 109 stores, 115 loads, 0 slot moves, 53 rematerialized; 26 slots (188 bytes)
      sink: 601 instructions; 43 register moves, 112 stores, 118 loads, 0 slot moves, 54 rematerialized; 26 slots (188 bytes)
    x86_64, 3 registers conv/0: sink moves 14 (rev1), reverse 119
      source: 130 instructions; 31 register moves, 67 stores, 69 loads, 0 slot moves, 22 rematerialized; 26 slots (168 bytes)
      sink: 130 instructions; 28 register moves, 66 stores, 68 loads, 0 slot moves, 20 rematerialized; 26 slots (168 bytes) |}]
