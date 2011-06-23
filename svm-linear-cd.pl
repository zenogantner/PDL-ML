#!/usr/bin/perl

# SVM example - coordinate descent for linear SVMs

# Get example dataset with
# wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale

# (c) 2011 Zeno Gantner
# License: GPL 3 or later

# TODO
#  - bias handling via features (for all linear SVM programs)

use strict;
use warnings;
use 5.10.1;

use English qw( -no_match_vars );
use Getopt::Long;
use List::Util;
use PDL;
use PDL::LinearAlgebra;
use PDL::NiceSlice;

GetOptions(
	   'help'              => \(my $help            = 0),
	   'verbose'           => \(my $verbose         = 0),
	   'compute-fit'       => \(my $compute_fit     = 0),
	   'epsilon=f'         => \(my $epsilon         = 0.1),
	   'training-file=s'   => \(my $training_file   = ''),
	   'test-file=s'       => \(my $test_file       = ''),
	   'prediction-file=s' => \(my $prediction_file = ''),
	   'c=f'               => \(my $c               = 1),
	  ) or usage(-1);

usage(0) if $help;


if ($training_file eq '') {
        say "Please give --training-file=FILE";
        usage(-1);
}

my ( $instances, $targets ) = convert_to_pdl(read_data($training_file));
my $num_instances = (dims $instances)[0];
my $num_features  = (dims $instances)[1];

# solve optimization problem
my ($alpha, $beta) = coordinate_descent($instances, $targets);
# prepare prediction function
my $predict = sub { return $beta x $_[0] >= 0 ? +1 : -1; };
my $predict_several = sub {
        my ($instances) = @_;        
        my $num_instances = (dims $instances)[0];
        
        my $predictions = zeros($num_instances);
        for (my $i = 0; $i < $num_instances; $i++) {
                $predictions($i) .= $beta x $instances($i) >= 0 ? +1 : -1;
        }
        
        return $predictions;
};

exit 0;

sub bounded_by {
        my ($x, $a, $b) = @_;
        
        return $a if $x <= $a;
        return $b if $x >= $b;
        return $x;
}

# solve dual optimization problem
sub coordinate_descent {
        my ($x, $y) = @_;

        my $alpha = zeros($num_instances);
        my $beta  = zeros($num_features);

        my $changes_counter = 0;
        do {
                $changes_counter = 0;
                # TODO random order
                foreach my $i (0 .. $num_instances - 1) {
                        my $delta_alpha = bounded_by(
                                                 (1 - $y($i) * $beta x $x($i)) / sum($x($i) * $x($i)),
                                                 -$alpha($i),
                                                 $c - $alpha($i),
                                                  );
                        
                        $alpha($i) .= $alpha($i) + $delta_alpha;
                        $beta = $beta + $delta_alpha * $y($i) * $x($i)->transpose;

                        $changes_counter++ if abs($delta_alpha) > $epsilon;
                }
                say "$changes_counter changes" if $verbose;
        } while $changes_counter;
        
        return ($alpha, $beta);
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

Perl Data Language SVM example: coordinate descent for linear SVMs

usage: $PROGRAM_NAME [OPTIONS] [INPUT]

    --help                  display this usage information
    --verbose               show diagnostic/progress output
    --epsilon=NUM           set convergence sensitivity to NUM
    --compute-fit           compute error on training data
    --training-file=FILE    read training data from FILE
    --test-file=FILE        evaluate on FILE
    --prediction-file=FILE  write predictions for instances in the test file to FILE
    --c=NUM                 complexity parameter C
END
    exit $return_code;
}
