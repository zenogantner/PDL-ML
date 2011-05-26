#!/usr/bin/perl

# SVM example - SMO variant worst-violating pair by Keerthi (2001)

# Get example dataset with
# wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale

# (c) 2011 Zeno Gantner
# License: GPL

# TODO:
#  - handle bias correctly
#  - handle arbitrary two-class and multi-class problems
#  - move shared code into module
#  - use sparse data structures
#  - shrinking heuristics
#  - --verbose
#  - other learning algorithms
#  - support-vector regression
#  - probability via Platt smoothing
#  - internal CV
#  - load/save model

use strict;
use warnings;
use 5.10.1;

use English qw( -no_match_vars );
use Getopt::Long;
use List::Util qw();
use PDL;
use PDL::LinearAlgebra;
use PDL::NiceSlice;

GetOptions(
	   'help'              => \(my $help            = 0),
	   'compute-fit'       => \(my $compute_fit     = 0),
	   'epsilon=f'         => \(my $epsilon         = 0.001),
	   'training-file=s'   => \(my $training_file   = ''),
	   'test-file=s'       => \(my $test_file       = ''),
	   'prediction-file=s' => \(my $prediction_file = ''),
	   'kernel=s'          => \(my $kernel          = 'rbf'),
	   'degree=i'          => \(my $degree          = 2),
	   'gamma=f'           => \(my $gamma           = 1),
	   'c=f'               => \(my $c               = 1),
	   #'probabilities'     => \(my $probabilities   = 0),
	  ) or usage(-1);

usage(0) if $help;


if ($training_file eq '') {
        say "Please give --training-file=FILE";
        usage(-1);
}

my %kernel = (
        'linear'     => sub { sum($_[0] * $_[1]) }, # TODO why does inner() not work?
        'polynomial' => sub { (1 + sum($_[0] * $_[1])) ** $degree },
        'rbf'        => sub { exp( (sum(($_[0] - $_[1]) * ($_[0] - $_[1])) / $gamma) ) }, # TODO think about possible speed-ups
);
my $K = $kernel{$kernel};


my ( $instances, $targets ) = convert_to_pdl(read_data($training_file));
say "X $instances";
say "y $targets";
my $num_instances = (dims $instances)[0];
my $num_features  = (dims $instances)[1];

# solve optimization problem
my $alpha = smo($instances, $targets);
# prepare prediction function
my $num_support_vectors = sum($alpha != 0);
my $relevant_instances       = zeros($num_support_vectors, $num_features);
my $relevant_instances_alpha = zeros($num_support_vectors); # actually: <alpha, y>
my $offset = 0; # TODO
my $pos = 0;
say "alpha $alpha";
say $relevant_instances_alpha;
for (my $i = 0; $i < $num_instances; $i++) {       
        if ($alpha($i) > 0) {
            $relevant_instances($pos)       .= $instances($i);
            $relevant_instances_alpha($pos) .= $alpha($i) * $targets($i);
            $pos++;
        }
}
print "ri ";
say $relevant_instances;
print "ri_a ";
say $relevant_instances_alpha;
my $predict = sub {
        my ($x) = @_;
        
        my $score = $offset;
        for (my $i = 0; $i < $num_support_vectors; $i++) {
                $score += $relevant_instances_alpha($i) * &$K($relevant_instances($i), $x);
        }
        
        return $score <=> 0;
};
my $predict_several = sub {
        my ($instances) = @_;        
        my $num_instances = (dims $instances)[0];
        
        my $predictions = zeros($num_instances);
        for (my $i = 0; $i < $num_instances; $i++) {
                $predictions($i) .= &$predict($instances($i));
        }
        
        return $predictions;
};

# compute fit
if ($compute_fit) {
        my $pred = &$predict_several($instances);

        my $fit_err  = sum(abs($pred - $targets));
        $fit_err /= $num_instances;

        say "FIT_ERR $fit_err N $num_instances";
}

# test/write out predictions
if ($test_file) {
        my ( $test_instances, $test_targets ) = convert_to_pdl(read_data($test_file));
        my $test_pred = &$predict_several($test_instances);

        if ($prediction_file) {
                write_vector($test_pred, $prediction_file);
        }
        else {
                my $num_test_instances = (dims $test_instances)[0];

                my $test_err  = sum(abs($test_pred - $test_targets));
                $test_err /= $num_test_instances;
                say "ERR $test_err N $num_test_instances";
        }
}

exit 0;

# TODO better name?
sub min_max {
        my ($x, $a, $b) = @_;
        
        return $x if $x > $a && $x < $b;
        return $a if $a >= $x && $a < $b;
        return $b;
}

sub max {
        my ($x, $y) = @_;
        
        return $x if $x > $y;
        return $y;
}

sub min {
        my ($x, $y) = @_;
        
        return $x if $x < $y;
        return $y;
}

# solve dual optimization problem
sub smo {
        my ($x, $y) = @_;

        my $alpha = zeros($num_instances);
        my $f     = $y->copy;

        my $i = 0;
        my $j = 0;
        for (my $idx = 0; $idx < $num_instances; $idx++) { # TODO find a more elegant PDL formulation
                if ($y($idx) == 1) {
                        $i = $idx;
                        last;
                }
        }        
        for (my $idx = 0; $idx < $num_instances; $idx++) { # TODO find a more elegant PDL formulation
                if ($y($idx) == -1) {
                        $j = $idx;
                        last;
                }
        }

        while ($f($i) - $f($j) > $epsilon) {
                say 'stopping criterion: ' . sum($f($i) - $f($j));
                #say "a[$i]=" . $alpha($i) . ", a[$j]=" . $alpha($j);
                #say "f[$i]=" . $f($i)     . ", f[$j]=" . $f($j);
                
                my $delta_alpha = ($f($i) - $f($j)) / ( &$K($x($i), $x($i)) +  &$K($x($j), $x($j)) - 2 * &$K($x($i), $x($j)) );
                # TODO cache/memoize kernel evaluation
                
                if ($y($i) * $y($j) == -1) {
                        $delta_alpha = $y($i) * min_max(
                                                        $y($i) * $delta_alpha,
                                                        - min($alpha($i), $alpha($j)),
                                                        $c - max($alpha($i), $alpha($j))
                                                        );
                }
                else {
                        $delta_alpha = $y($i) * min_max(
                                                        $y($i) * $delta_alpha,
                                                        - min($alpha($i), $c - $alpha($j)),
                                                        min($c - $alpha($i), $alpha($j))
                                                        );                        
                }
                $alpha($i) .= $alpha($i) + $y($i) * $delta_alpha; # TODO in-place modification?
                $alpha($j) .= $alpha($j) - $y($j) * $delta_alpha; # TODO in-place modification?
                
                my $max_f_index = 0;
                my $min_f_index = 0;
                for (my $k = 0; $k < $num_instances; $k++) {
                        $f($k) .= $f($k) - $delta_alpha * ( &$K($x($k), $x($i)) - &$K($x($k), $x($j)) );  # TODO in-place modification?
                        if ( ($y($k) == +1 && $alpha($k) < $c) || ($y($k) == -1 && $alpha($k) > 0) ) {
                                $max_f_index = $k if $f($k) > $f($max_f_index);
                        }
                        if ( ($y($k) == -1 && $alpha($k) < $c) || ($y($k) == +1 && $alpha($k) > 0) ) {
                                $min_f_index = $k if $f($k) < $f($min_f_index);
                        }
                }
                $i = $max_f_index;
                $j = $min_f_index;
        }

        return $alpha;
}

# convert Perl data structure to piddles
sub convert_to_pdl {
        my ($data_ref, $num_features) = @_;

        my $instances = zeros scalar @$data_ref, $num_features;
        my $targets   = zeros scalar @$data_ref; # TODO handle multi-class/multi-label here

        for (my $i = 0; $i < scalar @$data_ref; $i++) {
                my ($feature_value_ref, $target) = @{ $data_ref->[$i] };

                $targets($i) .= $target;

                foreach my $id (keys %$feature_value_ref) {
                        $instances($i, $id) .= $feature_value_ref->{$id};
                }
        }

        return ( $instances, $targets );
}

# read LIBSVM-formatted data from file
sub read_data {
        my ($training_file) = @_;

        my @labeled_instances = ();

        my $num_features = 0;

        open my $fh, '<', $training_file;
        while (<$fh>) {
                my $line = $_;
                chomp $line;

                my @tokens = split /\s+/, $line;
                my $label = shift @tokens;               
                $label = -1 if $label == 0;
                
                die "Label must be 1/0/-1, but is $label\n" if $label != -1 && $label != 1;

                my %feature_value = map { split /:/ } @tokens;
                $num_features = List::Util::max(keys %feature_value, $num_features);

                push @labeled_instances, [ \%feature_value, $label ];
        }
        close $fh;

        $num_features++; # take care of features starting index 0

        return (\@labeled_instances, $num_features); # TODO named return
}

# write row vector to text file, one line per entry
sub write_vector {
        my ($vector, $filename) = @_;
        open my $fh, '>', $filename;
        foreach my $col (0 .. (dims $vector)[0] - 1) {
                say $fh $vector->at($col, 0);
        }
        close $fh;
}


sub usage {
    my ($return_code) = @_;

    print << "END";
$PROGRAM_NAME

Perl Data Language SVM example

usage: $PROGRAM_NAME [OPTIONS] [INPUT]

    --help                  display this usage information
    --epsilon=NUM           set convergence sensitivity to NUM
    --compute-fit           compute error on training data
    --training-file=FILE    read training data from FILE
    --test-file=FILE        evaluate on FILE
    --prediction-file=FILE  write predictions for instances in the test file to FILE
    --kernel=linear|polynomial|rbf
    --gamma                 gamma parameter for the RBF (Gaussian) kernel
    --degree=INT            degree for the polynomial kernel (>0)
    --c=NUM                 complexity parameter C
END
    exit $return_code;
}
