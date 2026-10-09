open Err.Syntax

type error = [ `Source_io of string * string ]

let pp_error ppf (`Source_io (path, message)) =
  Fmt.pf ppf "cannot read source %S: %s" path message

let map_file path =
  match
    let fd = Unix.openfile path [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
    Fun.protect
      ~finally:(fun () -> Unix.close fd)
      (fun () ->
        let size = (Unix.fstat fd).Unix.st_size in
        if size = 0 then
          Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0
        else
          Bigarray.array1_of_genarray
            (Unix.map_file fd Bigarray.char Bigarray.c_layout false [| -1 |]))
  with
  | bytes -> Err.return bytes
  | exception Unix.Unix_error (e, _, _) ->
      Err.fail (`Source_io (path, Unix.error_message e))
  | exception Sys_error m -> Err.fail (`Source_io (path, m))

let sources_in_dir (doc : Pt2_checkpoint_map.Document.t) ~dir =
  let names =
    List.map
      (fun (s : Pt2_checkpoint_map.Document.Source.t) -> s.pin.name)
      doc.checkpoint_files
    @ match doc.graph_owned with Some p -> [ p.name ] | None -> []
  in
  Err.List.map
    (fun name ->
      let+ bytes = map_file (Filename.concat dir name) in
      { Pt2_checkpoint_map.Prepare.name; bytes })
    names
