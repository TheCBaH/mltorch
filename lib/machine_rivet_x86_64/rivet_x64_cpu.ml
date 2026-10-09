(* Whether this CPU can run what an artifact says it needs, answered from the
   kernel's own report: the flags of the first processor in [/proc/cpuinfo]. A
   machine whose flags cannot be read is refused, not assumed. AVX needs the OS
   to enable the register state as well, which the [avx] flag alone does not
   say; the kernel clears the flag when it has not. *)

module Feature = Machine_ir.Mir_target.Feature

(* The flag Linux's [/proc/cpuinfo] reports for a feature. *)
let flag = function
  | Feature.Avx -> Some "avx"
  | Feature.Avx2 -> Some "avx2"
  | Feature.Fma -> Some "fma"
  | Feature.Fp | Feature.Neon -> None
  | Feature.Sse2 -> Some "sse2"
  | Feature.Sse41 -> Some "sse4_1"

let flags_line path =
  match open_in path with
  | exception Sys_error _ -> None
  | ic ->
      let rec go () =
        match input_line ic with
        | exception End_of_file -> None
        | line -> (
            match String.index_opt line ':' with
            | Some i when String.trim (String.sub line 0 i) = "flags" ->
                Some
                  (String.split_on_char ' '
                     (String.trim
                        (String.sub line (i + 1) (String.length line - i - 1))))
            | _ -> go ())
      in
      let r = go () in
      close_in ic;
      r

(* The first feature the artifact needs and this CPU lacks, if any. *)
let admit ?(cpuinfo = "/proc/cpuinfo") features =
  match flags_line cpuinfo with
  | None -> Error Rivet_x64_refusal.Cpu_unknown
  | Some have ->
      List.fold_left
        (fun acc f ->
          match acc with
          | Error _ -> acc
          | Ok () -> (
              match flag f with
              | None -> Error (Rivet_x64_refusal.Cpu_feature (Feature.name f))
              | Some fl ->
                  if List.mem fl have then Ok ()
                  else Error (Rivet_x64_refusal.Cpu_feature (Feature.name f))))
        (Ok ()) features
