(* A LOADER that runs the unit compiled by the host compiler. *)

type loaded = { lib : Dl.library; entry : char Ctypes.ptr -> int64 }
type error = string

let pp_error = Fmt.string

let dir =
  lazy
    (let d = Loop_c_exec.Proc.temp_dir "gcc_loader" in
     at_exit (fun () -> Loop_c_exec.Proc.remove_tree d);
     d)

let counter = ref 0

let load ~host_symbols source =
  let unknown =
    List.filter
      (fun n -> not (List.mem n Loop_ir.Loop_c_dialect.host_symbols))
      host_symbols
  in
  if unknown <> [] then
    Error ("unknown host symbols: " ^ String.concat " " unknown)
  else begin
    incr counter;
    let dir = Lazy.force dir in
    let c = Filename.concat dir (Printf.sprintf "u%d.c" !counter) in
    let so = Filename.concat dir (Printf.sprintf "u%d.so" !counter) in
    Loop_c_exec.Proc.write_file c source;
    match
      Loop_c_exec.Proc.run
        (Loop_c_exec.Host.default_compiler
        @ [ "-shared"; "-fPIC"; c; "-o"; so; "-lm" ])
    with
    | Error m -> Error m
    | Ok (Loop_c_exec.Proc.Exited 0, _) ->
        let lib = Dl.dlopen ~filename:so ~flags:[ Dl.RTLD_NOW ] in
        let entry =
          Foreign.foreign ~from:lib "entry"
            Ctypes.(ptr char @-> returning int64_t)
        in
        Ok { lib; entry }
    | Ok (_, log) -> Error log
  end

let call t ~io = Ok (t.entry (Ctypes.bigarray_start Ctypes.array1 io))
let close t = Dl.dlclose ~handle:t.lib
