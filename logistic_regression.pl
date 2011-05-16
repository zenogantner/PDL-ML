#!/usr/bin/perl

# Machine learning examples

# Get example datasets for regression and classification with
# wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale

# (c) 2011 Zeno Gantner
# License: GPL

# TODO:
#  - create evaluation and prediction subroutines
#  - handle arbitrary two-class and multi-class problems
#  - --verbose
#  - SVM

use strict;
use warnings;
use 5.10.0;

use English qw( -no_match_vars );
use Getopt::Long;
use List::Util;
use PDL;
use PDL::LinearAlgebra;
use PDL::NiceSlice;

GetOptions(
	   'help'              => \(my $help            = 0),
	   'compute-fit'       => \(my $compute_fit     = 0),
#	   'shrinkage=f'       => \(my $shrinkage       = 0.0),
	   'epsilon=f'         => \(my $epsilon         = 0.001),
	   'training-file=s'   => \(my $training_file   = ''),
	   'test-file=s'       => \(my $test_file       = ''),
	   'prediction-file=s' => \(my $prediction_file = ''),
	  ) or usage(-1);

usage(0) if $help;

#$shrinkage += 0.0; # workaround for PDL or Getopt::Long bug (?)

if ($training_file eq '') {
        say "Please give --training-file=FILE";
        usage(-1);
}

my ( $instances, $targets ) = convert_to_pdl(read_data($training_file));

my $params = irls($instances, $targets);

# compute RSS and RMSE
# TODO compute accuracy
if ($compute_fit) {
        my $num_instances = (dims $instances)[0];

        my $pred = $params->transpose x $instances; # parentheses or OO notation are important here
        my $rss = sum(($pred - $targets) ** 2);
        my $rmse = sqrt($rss / $num_instances);
        say "RSS $rss FIT_RMSE $rmse N $num_instances";
}

# test/write out predictions
# TODO write out decisions
if ($test_file) {
        my ( $test_instances, $test_targets ) = convert_to_pdl(read_data($test_file));
        my $test_pred = $params->transpose x $test_instances;

        if ($prediction_file) {
                write_vector($test_pred, $prediction_file);
        }
        else {
                my $num_test_instances = (dims $test_instances)[0];

                my $test_rss  = sum(($test_pred - $targets) ** 2);
                my $test_rmse = sqrt($test_rss / $num_test_instances);
                say "RMSE $test_rmse N $num_test_instances";
        }
}

exit 0;

# compute logistic regression parameters using iteratively reweighted least squares (IRLS)
sub irls {
        # TODO add regularization
        my ($instances, $targets) = @_;

        my $num_instances = (dims $instances)[0];
        my $num_features  = (dims $instances)[1];
        my $params      = zeros(1, $num_features);
        my $old_p       = ones (1, $num_instances);
        my $delta;

        do {
                my $scores = $instances->transpose x $params;

                my $p = 1 / (1 + exp(-1 x $scores));

                my $w = $p * (1 - $p);

                #my $w_diag = stretcher($w);
                # ugly workaround
                my $w_diag = zeros($num_instances, $num_instances);
                for (my $i = 0; $i < $num_instances; $i++) {
                        $w_diag($i, $i) .= $w(0, $i);
                }

                my $w_diag_inv = minv $w_diag;

                my $z = $instances->transpose x $params + $w_diag_inv x ($targets->transpose - $p);
                my $xtw = $instances x $w_diag;

                $params = msolve( $xtw x $instances->transpose, $xtw x $z );

                $delta = sum(abs($p - $old_p));
                $old_p = $p->copy;
        } while ($delta > $epsilon);

        return  $params;
}

# convert Perl data structure to piddles
sub convert_to_pdl {
        my ($data_ref, $num_features) = @_;

        my $instances = zeros scalar @$data_ref, $num_features + 1;
        my $targets   = zeros scalar @$data_ref, 1;

        for (my $i = 0; $i < scalar @$data_ref; $i++) {
                my ($feature_value_ref, $target) = @{ $data_ref->[$i] };

                $instances($i, 0) .= 1; # this is the bias term
                $targets($i, 0) .= $target;

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

                my %feature_value = map { split /:/ } @tokens;
                $num_features = List::Util::max(keys %feature_value, $num_features);

                push @labeled_instances, [ \%feature_value, $label ];
        }
        close $fh;

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

Perl Data Language ridge regression example

usage: $PROGRAM_NAME [OPTIONS] [INPUT]

    --help                  display this usage information
    --compute-fit           compute RSS and RMSE on training data
    --training-file=FILE    read training data from FILE
    --test-file=FILE        evaluate on FILE
    --prediction-file=FILE  write predictions for instances in the test file to FILE
END
    exit $return_code;
}
