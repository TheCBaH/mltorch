(** Running external programs without a shell: the argument vector is passed to
    the kernel as is, so a path with spaces or metacharacters is one argument.
*)

type status = Exited of int | Signaled of int

val pp_status : Format.formatter -> status -> unit

val run : ?cwd:string -> string list -> (status * string, string) result
(** [run argv] runs [argv] to completion, capturing stdout and stderr together
    (into a file, so a chatty child cannot block on a full pipe) and returning
    them as text. [Error] only if the program could not be started. *)

val temp_dir : string -> string
(** A fresh directory under the system temporary directory. *)

val write_file : string -> string -> unit
val read_file : string -> string

val remove_tree : string -> unit
(** Removes a file or a directory with everything below it; silent if absent. *)
