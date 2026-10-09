(* What identifies an image, and what admits it to this CPU. *)

module F = Native_test.Graph_fixtures
module T = Machine_model_test.Model_test
module H = Machine_rivet_x86_64.Rivet_x64_host
module Mf = Machine_rivet_x86_64.Rivet_x64_manifest
module Rt = Machine_rivet_x86_64.Rivet_x64_route
module Rn = Machine_rivet_x86_64.Rivet_x64_runtime
module Cpu = Machine_rivet_x86_64.Rivet_x64_cpu
module Refusal = Machine_rivet_x86_64.Rivet_x64_refusal
module Feature = Machine_ir.Mir_target.Feature

let keys ?allocation ?runtime ?(pipeline = Ssa_backends.Pipeline.Exact) graph =
  let host =
    Result.get_ok
      (H.prepare ?allocation ?runtime ~pipeline (T.bundle (graph ())))
  in
  List.map Mf.key (H.manifests host)

let%expect_test "a manifest is stable and tells images apart" =
  let same a b = if a = b then "same" else "different" in
  let base = keys F.chain in
  Fmt.pr "again: %s@." (same base (keys F.chain));
  Fmt.pr "scanned allocation: %s@."
    (same base (keys ~allocation:Rt.Allocation.Scanned F.chain));
  Fmt.pr "representation pipeline: %s@."
    (same base (keys ~pipeline:Ssa_backends.Pipeline.Representation F.chain));
  Fmt.pr "libm mode: %s@." (same base (keys ~runtime:Rn.System_libm F.chain));
  [%expect
    {|
    again: same
    scanned allocation: different
    representation pipeline: different
    libm mode: different |}]

let%expect_test "a manifest names what an image needs" =
  let host =
    Result.get_ok
      (H.prepare ~runtime:Rn.System_libm ~pipeline:Ssa_backends.Pipeline.Exact
         (T.bundle
            (F.build "softmax"
               Graph_builder.(
                 let* x = input ~shape:(F.s 1 1 1 1 2 3) () in
                 softmax { Reduce.Softmax.axis = Axis.C } x))))
  in
  let m = List.hd (H.manifests host) in
  print_string
    (String.concat "\n"
       (List.filter
          (fun l -> not (String.length l >= 5 && String.sub l 0 5 = "code="))
          (String.split_on_char '\n' (Mf.to_string m))));
  [%expect
    {|
    target=x86_64
    source=Intel SDM 325462 084
    features=sse2
    runtime=system_libm
    binding=table
    helpers=exp
    owned=
    region=mir_region_0,24,16,bound
    region=mir_region_1,24,16,bound
    region=mir_region_200000,8,16,bss
    region=mir_region_200001,8,16,bss
    region=mir_region_1000000,104,16,bss
    planning.subject=cb3086cf512ae2a28e96ae8d18d790ce
    planning.policy=reference_f64
    planning.schedule=scalar
    planning.precision=f64
    planning.lanes=1
    planning.fma=forbidden
    planning.capabilities= |}]

let write contents =
  let path = Filename.temp_file "cpuinfo" "" in
  let oc = open_out path in
  output_string oc contents;
  close_out oc;
  path

let%expect_test "a CPU lacking a feature, or unreadable, is refused" =
  let show = function
    | Ok () -> "admitted"
    | Error r -> Fmt.str "%a" Refusal.pp r
  in
  let with_flags flags =
    write (Fmt.str "processor\t: 0\nflags\t\t: %s\n" flags)
  in
  Fmt.pr "sse2, needs sse2: %s@."
    (show (Cpu.admit ~cpuinfo:(with_flags "fpu sse sse2") [ Feature.Sse2 ]));
  Fmt.pr "sse2 only, needs fma: %s@."
    (show
       (Cpu.admit ~cpuinfo:(with_flags "sse2") [ Feature.Sse2; Feature.Fma ]));
  Fmt.pr "sse4.1 reported as sse4_1: %s@."
    (show (Cpu.admit ~cpuinfo:(with_flags "sse2 sse4_1") [ Feature.Sse41 ]));
  Fmt.pr "an AArch64 feature: %s@."
    (show (Cpu.admit ~cpuinfo:(with_flags "sse2") [ Feature.Neon ]));
  Fmt.pr "unreadable: %s@."
    (show (Cpu.admit ~cpuinfo:"/nonexistent/cpuinfo" [ Feature.Sse2 ]));
  Fmt.pr "this machine, sse2: %s@." (show (Cpu.admit [ Feature.Sse2 ]));
  [%expect
    {|
    sse2, needs sse2: admitted
    sse2 only, needs fma: this CPU does not report feature fma
    sse4.1 reported as sse4_1: admitted
    an AArch64 feature: this CPU does not report feature neon
    unreadable: this CPU's features cannot be read
    this machine, sse2: admitted |}]

(* Under the default mode exp is the image's own code: no dependency is
   declared, and the digest of that code is part of what the key covers. *)
let%expect_test "an owned helper is carried, not declared" =
  let softmax runtime =
    let host =
      Result.get_ok
        (H.prepare ~runtime ~pipeline:Ssa_backends.Pipeline.Exact
           (T.bundle
              (F.build "softmax"
                 Graph_builder.(
                   let* x = input ~shape:(F.s 1 1 1 1 2 3) () in
                   softmax { Reduce.Softmax.axis = Axis.C } x))))
    in
    List.hd (H.manifests host)
  in
  let m = softmax Rn.Dependency_free in
  Fmt.pr "declared: [%s]@." (String.concat "," m.Mf.helpers);
  Fmt.pr "carried: [%s]@." (String.concat "," (List.map fst m.Mf.owned));
  Fmt.pr "digest: %d hex digits@." (String.length (snd (List.hd m.Mf.owned)));
  Fmt.pr "differs from the libm image: %b@."
    (Mf.key m <> Mf.key (softmax Rn.System_libm));
  [%expect
    {|
    declared: []
    carried: [exp]
    digest: 32 hex digits
    differs from the libm image: true |}]
