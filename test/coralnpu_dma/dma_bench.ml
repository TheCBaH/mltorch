(* [dma_bench.exe DIR CONFIGURATION SWEEP...]: runs each sweep on the named
   subsystem configuration and prints one line per case, from the kernel's own
   [mcycle] reads. A sweep is

   - [PATH ENGINE SIZES] ({!Dma_copy}), e.g. [sram-dtcm dma16 16,64,4096], or
     [sram-dtcm dma16 4096/256] for a chain of 256-byte descriptors: each
     case's cycles and bytes a cycle;
   - [overlap WORK LENGTH TRIPS] ({!Dma_overlap}): the workload alone, a
     LENGTH-byte SRAM-to-DTCM chain alone, then both at once;
   - [gemm ENGINE [GEOMETRY] PROLOGUE OUT BLOCKS W K] ({!Dma_gemm}): a
     DMA-fed matrix multiply, B resident without a GEOMETRY ([resident-a/N],
     [row-major-a] or [split-k/N]);
   - [direct ENGINE BLOCKS W K]: the same multiply with its operands in SRAM,
     no DMA.

   Exits 1 when a destination differs from what the transfers must leave. *)

module C = Dma_copy
module K = Peak_kernels
module O = Dma_overlap

let or_exit pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp e;
      exit 1

let memory = function
  | "dtcm" -> C.Memory.Dtcm
  | "sram" -> Sram
  | m -> invalid_arg ("dma_bench: no memory " ^ m)

let path s =
  match String.split_on_char '-' s with
  | [ src; dst ] -> { C.Path.src = memory src; dst = memory dst }
  | _ -> invalid_arg ("dma_bench: PATH is SRC-DST, not " ^ s)

let engine s =
  let after p =
    let n = String.length p in
    if String.length s > n && String.sub s 0 n = p then
      Some (int_of_string (String.sub s n (String.length s - n)))
    else None
  in
  match
    (after "dma", after "rvv-copy-m", after "rvv-read-m", after "rvv-write-m")
  with
  | Some b, _, _, _ -> (
      match Coral_dma.Beat.of_bytes b with
      | Some b -> C.Engine.Dma b
      | None -> invalid_arg ("dma_bench: no beat of " ^ s))
  | _, Some m, _, _ -> Rvv_copy m
  | _, _, Some m, _ -> Rvv_read m
  | _, _, _, Some m -> Rvv_write m
  | _ when s = "scalar-copy" -> Scalar_copy
  | _ -> invalid_arg ("dma_bench: no engine " ^ s)

let work = function
  | "alu" -> O.Work.Alu
  | "dtcm-read" -> Dtcm_read
  | "dtcm-write" -> Dtcm_write
  | "sram-read" -> Sram_read
  | w -> invalid_arg ("dma_bench: no workload " ^ w)

let case s =
  match String.split_on_char '/' s with
  | [ n ] -> C.Case.whole (int_of_string n)
  | [ n; p ] -> { C.Case.length = int_of_string n; piece = int_of_string p }
  | _ -> invalid_arg ("dma_bench: no case " ^ s)

let params = { Coral_layout.Params.stack = 2048L; staging = 0L }

(* Wall-clock seconds a run may take: the largest multiplies are millions of
   cycles, which the SoC simulator runs at a few thousand a second. *)
let deadline = 3600.

(* Builds and runs one kernel: its timings slot's bytes, or exit 1. *)
let execute ?(matrix = false) dir p ~name ~items ~slots ~inputs ~timings ~dst
    ~expected =
  let extensions = if matrix then [ or_exit Fmt.string K.zvt ] else [] in
  let m =
    or_exit Fmt.string
      (K.to_cli
         (Err.bind (items ()) (fun items ->
              K.typed_module ~name:"dma" ~extensions items K.rv32_profile)))
  in
  let image =
    or_exit Coral_image.pp_error
      (Err.payload
         (Coral_image.build ~params ~rv32_profile:K.rv32_profile ~matrix p
            ~entry:K.entry ~kernel:[ m ] slots))
  in
  let inputs =
    or_exit Coral_dma.pp_error (Err.payload (inputs image.Coral_image.layout))
    |> Array.of_list
  in
  match
    Err.payload
      (Coral_soc_run.run ~deadline ~read_back:(Option.to_list timings) ~dir
         image ~bound:(fun k ->
           if k < Array.length inputs then Some inputs.(k) else None))
  with
  | Error e ->
      Fmt.pr "%s: %a@." name Coral_soc_run.Run_error.pp e;
      exit 1
  | Ok o ->
      if not (Coral_soc_run.matches o dst expected) then (
        Fmt.pr "%s: the destination is wrong@." name;
        exit 1);
      (Option.fold ~none:"" ~some:(fun k -> List.assoc k o.read_back) timings, o)

let word t k = Int32.to_int (String.get_int32_le t (4 * k))

let copy dir p (s : C.sweep) =
  let t, _ =
    execute dir p ~name:(C.name s) ~items:(C.items s) ~slots:(C.slots s)
      ~inputs:(C.inputs p s) ~timings:(Some C.timings_slot) ~dst:C.dst_slot
      ~expected:(C.expected s)
  in
  List.iteri
    (fun k (c : C.Case.t) ->
      let cycles = word t (2 * k) and status = word t ((2 * k) + 1) in
      Fmt.pr "%s %d%s: %d cycles, %.2f B/cycle%s@." (C.name s) c.length
        (if c.piece = c.length then "" else Printf.sprintf "/%d" c.piece)
        cycles
        (float_of_int c.length /. float_of_int cycles)
        (if status = 0 || status = Coral_dma.status_done then ""
         else Printf.sprintf ", status %08x" status))
    s.cases

(* The workload alone, the chain alone, then both at once. *)
let overlap dir p work length trips =
  let s =
    {
      O.work;
      length;
      cases =
        [
          { dma = false; trips };
          { dma = true; trips = 0 };
          { dma = true; trips };
        ];
    }
  in
  let t, _ =
    execute dir p ~name:(O.name s) ~items:(O.items s) ~slots:(O.slots s)
      ~inputs:(O.inputs p s) ~timings:(Some O.timings_slot) ~dst:O.dst_slot
      ~expected:(O.expected s)
  in
  let work_alone = word t 0 and dma_alone = word t 4 in
  let work_both = word t 6 and dma_both = word t 7 in
  Fmt.pr
    "%s x%d: work %d alone, %d beside the DMA; DMA %d alone, %d in all; %.0f%% \
     of the shorter hidden@."
    (O.name s) trips work_alone work_both dma_alone dma_both
    (100.
    *. float_of_int (work_alone + dma_alone - dma_both)
    /. float_of_int (min work_alone dma_alone))

(* The DMA's descriptor polling, case by case ({!Dma_poll}). *)
let poll dir p =
  let t, _ =
    execute dir p ~name:Dma_poll.name ~items:Dma_poll.items
      ~slots:Dma_poll.slots ~inputs:(Dma_poll.inputs p)
      ~timings:(Some Dma_poll.timings_slot) ~dst:Dma_poll.dst_slot
      ~expected:Dma_poll.expected
  in
  List.iteri
    (fun k (c : Dma_poll.Case.t) ->
      Fmt.pr "%s: stream %d, flag to done %d, CUR_DESC %08x, STATUS %x@." c.name
        (word t (4 * k))
        (word t ((4 * k) + 1))
        (word t ((4 * k) + 2))
        (word t ((4 * k) + 3)))
    Dma_poll.cases

module G = Dma_gemm

let gemm_engine = function
  | "rvv-bf16" -> G.Engine.Rvv_bf16
  | "rvv-f32" -> Rvv_f32
  | "rvv-i8" -> Rvv_i8
  | "zvt-f32" -> Zvt_f32
  | "zvt-i8" -> Zvt_i8
  | e -> invalid_arg ("dma_bench: no matrix multiply " ^ e)

let gemm dir p (g : G.t) =
  let t, (o : Coral_soc_run.outcome) =
    execute ~matrix:(G.Engine.matrix g.engine) dir p ~name:(G.name g)
      ~items:(G.items g) ~slots:(G.slots g) ~inputs:(G.inputs p g)
      ~timings:(Some G.Slot.timings) ~dst:G.Slot.c ~expected:(G.expected g)
  in
  Fmt.pr
    "%s: C as expected, %Ld cycles, %.1f ops/cycle; %d waiting for %s, %d for \
     the rest@."
    (G.name g) o.mcycle
    (float_of_int (G.ops g) /. Int64.to_float o.mcycle)
    (word t 1)
    (match g.geometry with
    | Resident_a _ -> "A and the first panel"
    | Resident_b | Row_major_a -> "B and the first block"
    | Split_k _ -> "B's first chunk and the first block")
    (word t 0)

(* The same multiply with A, B and C read in place from SRAM. *)
let direct dir p (g : G.t) =
  let k = G.direct g in
  let inputs = k.K.inputs in
  let out = List.length k.slots - 1 in
  let _, (o : Coral_soc_run.outcome) =
    execute ~matrix:k.matrix dir p
      ~name:(G.name g ^ " direct")
      ~items:k.items ~slots:k.slots
      ~inputs:(fun _ -> Ok inputs)
      ~timings:None ~dst:out ~expected:k.expected
  in
  Fmt.pr
    "%s M=%d N=%d K=%d, operands in SRAM, no DMA: C as expected, %Ld cycles, \
     %.1f ops/cycle@."
    (G.Engine.name g.engine)
    (G.rows g * g.blocks)
    g.width g.depth o.mcycle
    (float_of_int (G.ops g) /. Int64.to_float o.mcycle)

let gemm_prologue = function
  | "a-by-dma" | "b-by-dma" -> G.Prologue.Dma
  | "a-shared" | "b-shared" -> Shared
  | o -> invalid_arg ("dma_bench: no prologue " ^ o)

let gemm_out = function
  | "c-by-core" -> G.Out.Core
  | "c-by-dma" -> Dma
  | o -> (
      try Scanf.sscanf o "c-split-%d%!" (fun n -> G.Out.Split n)
      with Scanf.Scan_failure _ | End_of_file | Failure _ ->
        invalid_arg ("dma_bench: no way out " ^ o))

(* [resident-a/PANEL], [row-major-a] or [split-k/CHUNK], or resident B when
   the prologue comes first. *)
let gemm_geometry rest =
  let after prefix g =
    let n = String.length prefix in
    if String.starts_with ~prefix g then
      Some (int_of_string (String.sub g n (String.length g - n)))
    else None
  in
  match rest with
  | g :: rest' -> (
      match (after "resident-a/" g, after "split-k/" g) with
      | Some panel, _ -> (G.Geometry.Resident_a { panel }, rest')
      | _, Some chunk -> (Split_k { chunk }, rest')
      | None, None when g = "row-major-a" -> (Row_major_a, rest')
      | None, None -> (Resident_b, rest))
  | [] -> (Resident_b, rest)

let () =
  let dir = Sys.argv.(1) in
  let p =
    or_exit Coral_profile.pp_error
      (Err.payload (Coral_profile.load ~dir Sys.argv.(2)))
  in
  let rec sweeps = function
    | "direct" :: engine :: blocks :: width :: depth :: rest ->
        let engine = gemm_engine engine and width = int_of_string width in
        (* An RVV block's lanes are a panel of a wider W. *)
        let geometry =
          match G.Engine.lanes engine with
          | Some panel when panel < width -> G.Geometry.Resident_a { panel }
          | Some _ | None -> Resident_b
        in
        direct dir p
          {
            G.blocks = int_of_string blocks;
            depth = int_of_string depth;
            engine;
            geometry;
            out = Dma;
            prologue = Dma;
            width;
          };
        sweeps rest
    | "gemm" :: engine :: rest -> (
        match gemm_geometry rest with
        | geometry, prologue :: out :: blocks :: width :: depth :: rest ->
            gemm dir p
              {
                G.blocks = int_of_string blocks;
                depth = int_of_string depth;
                engine = gemm_engine engine;
                geometry;
                out = gemm_out out;
                prologue = gemm_prologue prologue;
                width = int_of_string width;
              };
            sweeps rest
        | _ ->
            invalid_arg
              "dma_bench: gemm ENGINE [GEOMETRY] PROLOGUE OUT BLOCKS W K")
    | "poll" :: rest ->
        poll dir p;
        sweeps rest
    | "overlap" :: w :: length :: trips :: rest ->
        overlap dir p (work w) (int_of_string length) (int_of_string trips);
        sweeps rest
    | pa :: en :: sizes :: rest ->
        copy dir p
          {
            C.path = path pa;
            engine = engine en;
            cases = List.map case (String.split_on_char ',' sizes);
          };
        sweeps rest
    | [] -> ()
    | _ -> invalid_arg "dma_bench: a sweep is PATH ENGINE SIZES"
  in
  sweeps (List.tl (List.tl (List.tl (Array.to_list Sys.argv))))
