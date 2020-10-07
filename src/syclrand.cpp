/*
TODO:
1) Create a SyCLRAND super class
   Internally, it will have a object from the RNG class. Seeding and generating
   random numbers are done through the RNG object
2) Every RNG is a derived class from the RNG class.
   Conceptually, every RNG has a seed value (or array of seed values) from the
   beginning. Individual RNGs have a state buffer and may have parameter
   tables, etc, that depend on the RNG algo. To achieve this, make every RNG as
   a class, and to interface with the SyCLRAND object, make the every RNG class
   a derived class from some super class, which we will call as SyCLRAND

*/