(* Rivet's encoding of a typed module against GNU's of the same module
   ([tamper] edits the printed source first: evidence that a difference is seen). The
   module is printed once by [Gnu_module]; GNU as assembles it and GNU ld links
   it with each section at the address Rivet bound it to. Equal loadable bytes
   and equal global symbol addresses mean the two assemblers read the module
   alike, including every fixup and relocation. A disagreement names the first
   byte. This cannot close a native execution gate: it says nothing about what
   the code does, only that two encoders agree. *)

open Asm_core
module P = Driver.Pipeline.Make (Aarch64)

module Verdict = struct
  type t =
    | Agree of { segments : int; bytes : int; symbols : int }
    | Differ of { segment : string; offset : int; rivet : int; gnu : int }
    | Symbols of { name : string; rivet : int64 option; gnu : int64 option }
    | Tool of { command : string; status : int; output : string }
    | Rivet of string

  let pp fmt = function
    | Agree { segments; bytes; symbols } ->
        Fmt.pf fmt "agree (%d segments, %d bytes, %d symbols)" segments bytes
          symbols
    | Differ { segment; offset; rivet; gnu } ->
        Fmt.pf fmt "%s+%d: rivet %02x, gnu %02x" segment offset rivet gnu
    | Symbols { name; rivet; gnu } ->
        let a fmt = function
          | Some v -> Fmt.pf fmt "%Lx" v
          | None -> Fmt.string fmt "absent"
        in
        Fmt.pf fmt "symbol %s: rivet %a, gnu %a" name a rivet a gnu
    | Tool { command; status; output } ->
        Fmt.pf fmt "%s exited %d: %s" command status (String.trim output)
    | Rivet m -> Fmt.pf fmt "rivet: %s" m
end

let syntax = { Gnu_module.type_char = '%' }

let assembly m =
  Gnu_module.to_string syntax ~instruction:Aarch64.Instruction.pp_gnu m

let write path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

let read path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

(* Runs a command, returning its combined output. *)
let run command =
  let out = Filename.temp_file "gnu" ".out" in
  let status =
    Sys.command (Printf.sprintf "%s >%s 2>&1" command (Filename.quote out))
  in
  let output = read out in
  Sys.remove out;
  if status = 0 then Ok output
  else Error (Verdict.Tool { command; status; output })

let ( let* ) = Result.bind

let rec remove_tree path =
  if Sys.is_directory path then begin
    Array.iter
      (fun f -> remove_tree (Filename.concat path f))
      (Sys.readdir path);
    Unix.rmdir path
  end
  else Sys.remove path

let in_temp_dir f =
  let dir = Filename.temp_file "gnu_coherence" "" in
  Sys.remove dir;
  Unix.mkdir dir 0o700;
  Fun.protect ~finally:(fun () -> remove_tree dir) (fun () -> f dir)

(* The loadable segments of the link, by name, at their bound addresses. *)
let gnu_segment dir elf (s : Image.segment) =
  let bin = Filename.concat dir ("seg" ^ s.Image.name ^ ".bin") in
  let* _ =
    run
      (Printf.sprintf "objcopy -O binary -j %s %s %s"
         (Filename.quote s.Image.name)
         (Filename.quote elf) (Filename.quote bin))
  in
  Ok (read bin)

let gnu_symbols elf =
  let* out =
    run (Printf.sprintf "nm -g --defined-only %s" (Filename.quote elf))
  in
  Ok
    (List.filter_map
       (fun line ->
         match String.split_on_char ' ' (String.trim line) with
         | [ addr; _; name ] -> (
             match Int64.of_string_opt ("0x" ^ addr) with
             | Some a -> Some (name, a)
             | None -> None)
         | _ -> None)
       (String.split_on_char '\n' out))

let first_difference a b =
  let n = min (String.length a) (String.length b) in
  let rec go i =
    if i >= n then
      if String.length a = String.length b then None else Some (i, 0, 0)
    else if a.[i] <> b.[i] then Some (i, Char.code a.[i], Char.code b.[i])
    else go (i + 1)
  in
  go 0

let base = 0x10_0000L
let stride = 0x100_0000L

let check ?(tamper = Fun.id) ~entry
    (modules : Aarch64.Instruction.t Normalized_ast.module_ list) : Verdict.t =
  let result =
    let lower m =
      match P.lower ~state:Aarch64.default_state m with
      | Ok l -> Ok l
      | Error e -> Error (Verdict.Rivet (Foundation.Diag.render e))
    in
    let* lowered =
      List.fold_right
        (fun m acc ->
          let* acc = acc in
          let* l = lower m in
          Ok (l :: acc))
        modules (Ok [])
    in
    let* laid =
      match
        match lowered with
        | [ one ] -> P.plan ~entry one
        | many -> P.plan_many ~entry many
      with
      | Ok l -> Ok l
      | Error e -> Error (Verdict.Rivet (Foundation.Diag.render e))
    in
    let plan = Image.plan_of laid in
    let addresses =
      List.mapi
        (fun i (s : Image.segment_plan) ->
          (s.Image.seg_name, Int64.add base (Int64.mul stride (Int64.of_int i))))
        plan.Image.segments
    in
    let* image =
      match Image.bind_image laid ~addresses with
      | Ok i -> Ok i
      | Error e -> Error (Verdict.Rivet (Foundation.Diag.render e))
    in
    in_temp_dir (fun dir ->
        let* objects =
          List.fold_left
            (fun acc (i, m) ->
              let* acc = acc in
              let s = Filename.concat dir (Printf.sprintf "m%d.s" i) in
              let o = Filename.concat dir (Printf.sprintf "m%d.o" i) in
              write s (tamper (assembly m));
              let* _ =
                run
                  (Printf.sprintf "as -o %s %s" (Filename.quote o)
                     (Filename.quote s))
              in
              Ok (o :: acc))
            (Ok [])
            (List.mapi (fun i m -> (i, m)) modules)
        in
        let elf = Filename.concat dir "out.elf" in
        let starts =
          String.concat " "
            (List.map
               (fun (n, a) -> Printf.sprintf "--section-start=%s=0x%Lx" n a)
               addresses)
        in
        let* _ =
          run
            (Printf.sprintf "ld -o %s -e %s %s %s" (Filename.quote elf)
               (Filename.quote entry) starts
               (String.concat " " (List.rev_map Filename.quote objects)))
        in
        let* bytes_checked =
          List.fold_left
            (fun acc (s : Image.segment) ->
              let* total = acc in
              if String.length s.Image.bytes = 0 then Ok total
              else
                let* gnu = gnu_segment dir elf s in
                match first_difference s.Image.bytes gnu with
                | None -> Ok (total + String.length gnu)
                | Some (offset, r, g) ->
                    Error
                      (Verdict.Differ
                         { segment = s.Image.name; offset; rivet = r; gnu = g }))
            (Ok 0) image.Image.segments
        in
        let* gnu_syms = gnu_symbols elf in
        let* () =
          List.fold_left
            (fun acc (name, addr) ->
              let* () = acc in
              match List.assoc_opt name gnu_syms with
              | Some a when Int64.equal a addr -> Ok ()
              | gnu -> Error (Verdict.Symbols { name; rivet = Some addr; gnu }))
            (Ok ()) image.Image.exports
        in
        Ok
          (Verdict.Agree
             {
               segments = List.length image.Image.segments;
               bytes = bytes_checked;
               symbols = List.length image.Image.exports;
             }))
  in
  match result with Ok v -> v | Error v -> v
