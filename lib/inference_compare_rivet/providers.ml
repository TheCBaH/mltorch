(* The Rivet AArch64 routes as variant words. Each word names one execution
   mechanism (the image route or the table host) with its allocator and
   runtime, and the identity is built from that configuration. The host's CPU
   feature report is read when the runner is made; a host that does not give
   one is not an AArch64 host, and the word refuses. *)

module P = Inference_compare_pt2
module Rt = Machine_rivet_aarch64.Rivet_a64_route
module Rn = Machine_rivet_aarch64.Rivet_a64_runtime
module Cpu = Machine_rivet_aarch64.Rivet_a64_cpu

let isa = "aarch64"

let features () =
  match Cpu.features_line "/proc/cpuinfo" with
  | Some f -> Ok f
  | None -> Error "no AArch64 CPU feature report (/proc/cpuinfo)"

let available () = Result.map ignore (features ())

type mechanism = Image | Table

let mechanism_name = function Image -> "image" | Table -> "table"

module Image_default = Image_engine.Make (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
end)

module Image_scanned = Image_engine.Make (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
end)

module Table_default = Table_engine.Make (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
end)

module Table_scanned = Table_engine.Make (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
end)

(* The same four routes with every kernel binary32, twice: [fp32] the planned
   [Fp32_ordered] pipeline for neon128 (the engine as deployed, vector loops
   where the policy keeps each output's order), [fp32s] the same policy
   scalar, so the two are compared exactly. *)
let fp32_pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.neon128;
    }

let fp32s_pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

module Image_fp32 = Image_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
  let pipeline = fp32_pipeline
end)

module Image_scanned_fp32 = Image_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
  let pipeline = fp32_pipeline
end)

module Table_fp32 = Table_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
  let pipeline = fp32_pipeline
end)

module Table_scanned_fp32 = Table_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
  let pipeline = fp32_pipeline
end)

module Image_fp32s = Image_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
  let pipeline = fp32s_pipeline
end)

module Image_scanned_fp32s = Image_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
  let pipeline = fp32s_pipeline
end)

module Table_fp32s = Table_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Reference
  let runtime = Rn.Dependency_free
  let pipeline = fp32s_pipeline
end)

module Table_scanned_fp32s = Table_engine.Make_pipeline (struct
  let allocation = Rt.Allocation.Scanned
  let runtime = Rn.Dependency_free
  let pipeline = fp32s_pipeline
end)

module Runners = struct
  module Image_fp32 = P.Graph_runner.Make (Image_fp32)
  module Image_scanned_fp32 = P.Graph_runner.Make (Image_scanned_fp32)
  module Table_fp32 = P.Graph_runner.Make (Table_fp32)
  module Table_scanned_fp32 = P.Graph_runner.Make (Table_scanned_fp32)
  module Image_fp32s = P.Graph_runner.Make (Image_fp32s)
  module Image_scanned_fp32s = P.Graph_runner.Make (Image_scanned_fp32s)
  module Table_fp32s = P.Graph_runner.Make (Table_fp32s)
  module Table_scanned_fp32s = P.Graph_runner.Make (Table_scanned_fp32s)
  module Image_default = P.Graph_runner.Make (Image_default)
  module Image_scanned = P.Graph_runner.Make (Image_scanned)
  module Table_default = P.Graph_runner.Make (Table_default)
  module Table_scanned = P.Graph_runner.Make (Table_scanned)
end

let identity ?(pipeline = Ssa_backends.Pipeline.Exact) ~name ~mechanism
    ~allocation ~runtime features =
  Compiled_identity.make ~engine:name
    ~route:(Printf.sprintf "rivet-%s-%s" (mechanism_name mechanism) isa)
    ~configuration:
      (String.concat "; "
         [
           (match pipeline with
           | Ssa_backends.Pipeline.Exact -> "scalar binary64"
           | Ssa_backends.Pipeline.Planned { target; _ }
             when target.Ssa_ir.Ssa_target.name
                  <> Ssa_ir.Ssa_target.scalar.Ssa_ir.Ssa_target.name ->
               "binary32 (fp32_ordered), vectors for "
               ^ target.Ssa_ir.Ssa_target.name
           | _ -> "scalar binary32 (fp32_ordered)");
           "isa=" ^ isa;
           "mechanism=" ^ mechanism_name mechanism;
           "allocation=" ^ Rt.Allocation.name allocation;
           "runtime=" ^ Rn.name runtime;
           "pipeline=" ^ Ssa_backends.Pipeline.name pipeline;
         ])
    ~environment:
      (String.concat ":"
         (Sys.os_type :: string_of_int Sys.word_size :: isa :: features))
    ~components:
      (Pins.rivet :: Pins.schedule
      :: (match mechanism with Image | Table -> [ Pins.selection ]))

let provider ?pipeline ~name ~mechanism ~allocation ~runtime create :
    P.Provider.t =
  {
    name;
    available;
    instantiate =
      (fun () ->
        Result.map
          (fun features ->
            let id =
              identity ?pipeline ~name ~mechanism ~allocation ~runtime features
            in
            {
              P.Provider.runner = create ~identity:(fun () -> id);
              identity = id;
            })
          (features ()));
  }

let providers : P.Provider.t list =
  [
    provider ~name:"rivet" ~mechanism:Image ~allocation:Rt.Allocation.Reference
      ~runtime:Rn.Dependency_free Runners.Image_default.create;
    provider ~name:"rivet-scanned" ~mechanism:Image
      ~allocation:Rt.Allocation.Scanned ~runtime:Rn.Dependency_free
      Runners.Image_scanned.create;
    provider ~name:"rivet-table" ~mechanism:Table
      ~allocation:Rt.Allocation.Reference ~runtime:Rn.Dependency_free
      Runners.Table_default.create;
    provider ~name:"rivet-table-scanned" ~mechanism:Table
      ~allocation:Rt.Allocation.Scanned ~runtime:Rn.Dependency_free
      Runners.Table_scanned.create;
    provider ~pipeline:fp32_pipeline ~name:"rivet-fp32" ~mechanism:Image
      ~allocation:Rt.Allocation.Reference ~runtime:Rn.Dependency_free
      Runners.Image_fp32.create;
    provider ~pipeline:fp32_pipeline ~name:"rivet-scanned-fp32" ~mechanism:Image
      ~allocation:Rt.Allocation.Scanned ~runtime:Rn.Dependency_free
      Runners.Image_scanned_fp32.create;
    provider ~pipeline:fp32_pipeline ~name:"rivet-table-fp32" ~mechanism:Table
      ~allocation:Rt.Allocation.Reference ~runtime:Rn.Dependency_free
      Runners.Table_fp32.create;
    provider ~pipeline:fp32_pipeline ~name:"rivet-table-scanned-fp32"
      ~mechanism:Table ~allocation:Rt.Allocation.Scanned
      ~runtime:Rn.Dependency_free Runners.Table_scanned_fp32.create;
    provider ~pipeline:fp32s_pipeline ~name:"rivet-fp32s" ~mechanism:Image
      ~allocation:Rt.Allocation.Reference ~runtime:Rn.Dependency_free
      Runners.Image_fp32s.create;
    provider ~pipeline:fp32s_pipeline ~name:"rivet-scanned-fp32s"
      ~mechanism:Image ~allocation:Rt.Allocation.Scanned
      ~runtime:Rn.Dependency_free Runners.Image_scanned_fp32s.create;
    provider ~pipeline:fp32s_pipeline ~name:"rivet-table-fp32s" ~mechanism:Table
      ~allocation:Rt.Allocation.Reference ~runtime:Rn.Dependency_free
      Runners.Table_fp32s.create;
    provider ~pipeline:fp32s_pipeline ~name:"rivet-table-scanned-fp32s"
      ~mechanism:Table ~allocation:Rt.Allocation.Scanned
      ~runtime:Rn.Dependency_free Runners.Table_scanned_fp32s.create;
  ]

let register () = List.iter P.Provider.register providers
