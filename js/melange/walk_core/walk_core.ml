(* The entry module a wrapped library would get for free.

   lib/walk_core is wrapped, so consumers say [Walk_core.Pcg]. The mirror cannot
   also be called [walk_core] -- two libraries of one name cannot coexist in a
   dune context, and `enabled_if` does not help because both are enabled under
   the melange profile. So the mirror is [walk_core_mel] and [wrapped false],
   which would otherwise publish [Pcg] and [Float32] as bare top-level modules
   and break every [Walk_core.] reference in the probe. This restores the
   qualified path by hand.

   Keep in step with lib/walk_core: a module added there must be listed here, or
   it is simply invisible to melange consumers. *)

module Float32 = Float32
module Limits = Limits
module Pcg = Pcg
module Shape = Shape
module Walk = Walk
module Window_math = Window_math
