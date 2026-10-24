(* [dma_bench.exe DIR CONFIGURATION SWEEP...]: runs each sweep on the named
   subsystem configuration and prints one line per case, from the kernel's own
   [mcycle] reads. A sweep is

   - [PATH ENGINE SIZES] ({!Dma_copy}), e.g. [sram-dtcm dma16 16,64,4096], or
     [sram-dtcm dma16 4096/256] for a chain of 256-byte descriptors: each
     case's cycles and bytes a cycle;
   - [overlap WORK LENGTH TRIPS] ({!Dma_overlap}): the workload alone, a
     LENGTH-byte SRAM-to-DTCM chain alone, then both at once.

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

(* Builds and runs one kernel: its timings slot's bytes, or exit 1. *)
let execute dir p ~name ~items ~slots ~inputs ~timings ~dst ~expected =
  let m =
    or_exit Fmt.string
      (K.to_cli
         (Err.bind (items ()) (fun items ->
              K.typed_module ~name:"dma" ~extensions:[] items K.rv32_profile)))
  in
  let image =
    or_exit Coral_image.pp_error
      (Err.payload
         (Coral_image.build ~params ~rv32_profile:K.rv32_profile p
            ~entry:K.entry ~kernel:[ m ] slots))
  in
  let inputs =
    or_exit Coral_dma.pp_error (Err.payload (inputs image.Coral_image.layout))
    |> Array.of_list
  in
  match
    Err.payload
      (Coral_soc_run.run ~read_back:[ timings ] ~dir image ~bound:(fun k ->
           if k < Array.length inputs then Some inputs.(k) else None))
  with
  | Error e ->
      Fmt.pr "%s: %a@." name Coral_soc_run.Run_error.pp e;
      exit 1
  | Ok o ->
      if not (Coral_soc_run.matches o dst expected) then (
        Fmt.pr "%s: the destination is wrong@." name;
        exit 1);
      List.assoc timings o.read_back

let word t k = Int32.to_int (String.get_int32_le t (4 * k))

let copy dir p (s : C.sweep) =
  let t =
    execute dir p ~name:(C.name s) ~items:(C.items s) ~slots:(C.slots s)
      ~inputs:(C.inputs p s) ~timings:C.timings_slot ~dst:C.dst_slot
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
  let t =
    execute dir p ~name:(O.name s) ~items:(O.items s) ~slots:(O.slots s)
      ~inputs:(O.inputs p s) ~timings:O.timings_slot ~dst:O.dst_slot
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

let () =
  let dir = Sys.argv.(1) in
  let p =
    or_exit Coral_profile.pp_error
      (Err.payload (Coral_profile.load ~dir Sys.argv.(2)))
  in
  let rec sweeps = function
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
